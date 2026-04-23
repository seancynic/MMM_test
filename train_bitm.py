import os
import warnings
import json
import torch
import torch.nn.functional as F
import numpy as np

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
from models.bitm import BiTMBERT
from dataset import dataset_TM_train, dataset_TM_eval, dataset_tokenize
from exit.utils import get_model, generate_src_mask, init_save_folder, maybe_data_parallel
from utils.eval_bitm import eval_bitm_t2m, eval_bitm_m2t

from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
device = torch.device('cuda')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

init_save_folder(args)

args.vq_dir = f'./output/vq/{args.vq_name}'
args.resume_pth = f'{args.vq_dir}/net_last.pth'
codebook_train_dir = f'{args.vq_dir}/codebook_train/'
codebook_val_dir = f'{args.vq_dir}/codebook_val/'
codebook_test_dir = f'{args.vq_dir}/codebook_test/'
os.makedirs(args.vq_dir, exist_ok=True)
os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(f'{args.out_dir}/html', exist_ok=True)
os.makedirs(codebook_train_dir, exist_ok=True)
os.makedirs(codebook_val_dir, exist_ok=True)
os.makedirs(codebook_test_dir, exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
wrapper_opt = get_opt(dataset_opt_path, device)
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- GloVe ---- #####
from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')

##### ---- BERT Tokenizer ---- #####
bert_name = 'google-bert/bert-large-uncased'
tokenizer = AutoTokenizer.from_pretrained(bert_name)
special_ids_t = {
    'mask_id': tokenizer.mask_token_id,
    'cls_id': tokenizer.cls_token_id,
    'eos_id': tokenizer.sep_token_id,
    'pad_id': tokenizer.pad_token_id
}

##### ---- VQ-VAE ---- #####
net = vqvae.HumanVQVAE(args,  # use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)
print('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.to(device)
net.eval()

special_ids_m = {
    'mask_id': args.nb_code + 2,
    'pad_id': args.nb_code + 1,
    'end_id': args.nb_code,
}

##### ---- Text2Motion Transformer ---- #####
bitm_model = BiTMBERT(bert_name=bert_name,
                      vqvae=net,
                      vocab_m=args.nb_code,
                      max_t=args.max_t,
                      max_m=args.max_m,
                      first_modality=args.first_modality,
                      dropout_rate=args.drop_out_rate,
                      motion_encoder_layers=args.motion_encoder_layers,
                      motion_decoder_layers=args.motion_decoder_layers)
bitm_model.save_state_dict()
if args.resume_trans is not None:
    print('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    bitm_model.load_state_dict(ckpt['bitm'], strict=True)
bitm_model.to(device)
bitm_model.train()
bitm_model, parallel_info = maybe_data_parallel(
    bitm_model,
    batch_size=args.batch_size,
    min_batch_per_gpu=args.min_batch_per_gpu,
    logger=logger
)
logger.info(
    f"Runtime parallelism: visible_gpus={parallel_info['visible_gpus']}, "
    f"used_gpus={parallel_info['used_gpus']}, "
    f"batch_per_gpu={parallel_info['batch_per_gpu']:.1f}, "
    f"data_parallel={parallel_info['data_parallel']}."
)

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, bitm_model, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- get codebook ---- #####
codebooks = {'train': codebook_train_dir, 'val': codebook_val_dir, 'test': codebook_test_dir}
for type, codebook_dir in codebooks.items():
    if len(os.listdir(codebook_dir)) == 0:
        dataloader_token = dataset_tokenize.DATALoaderNew(args.dataname, type=type, batch_size=1, unit_length=2 ** args.down_t)
        for batch in dataloader_token:
            pose, name = batch
            pose = pose.to(device).float()  # bs, nb_joints, joints_dim, seq_len
            target = net(pose, type='encode')
            target = target.cpu().numpy()
            np.save(os.path.join(codebook_dir, f'{name[0]}.npy'), target)

##### ---- Dataloader ---- #####
train_loader = dataset_TM_train.DATALoaderNew(args.dataname, codebook_train_dir, args.nb_code, args.batch_size, unit_length=2 ** args.down_t)
train_loader_iter = dataset_TM_train.cycle(train_loader)

##### ---- Evaluation ---- #####
def compute_result(pred_seq_masked, target, seq_mask_no_end):
    pred_seq_masked_index = pred_seq_masked.argmax(dim=-1)  # (num_valid,)
    target_seq_masked = torch.masked_select(target, seq_mask_no_end)  # (num_valid,)
    right_seq_masked = (pred_seq_masked_index == target_seq_masked).sum()  # compare with label

    return right_seq_masked

def get_acc(cls_pred, target, mask):
    # Only look at the indices where mask is True
    active_outputs = cls_pred[mask]
    active_targets = target[mask]

    # Get the predicted class indices (argmax is faster than max + softmax)
    predictions = active_outputs.argmax(dim=-1)

    # Calculate accuracy
    correct = (predictions == active_targets).float().sum()
    return (correct / mask.sum()) * 100

##### ---- Masking ---- #####
def masking(ids, seq_lens: torch.Tensor, batch_size, max_len, mask_id, probs: list, no_corruption=False):
    if args.pkeep == -1:
        proba = torch.rand(1, device=ids.device)
        mask = torch.rand(ids.shape, device=ids.device) < proba
    else:
        mask = torch.rand(ids.shape, device=ids.device) < args.pkeep

    # Step 1: Random only real token. To prevent pad token got mixed up.
    seq_mask_no_end = generate_src_mask(max_len, seq_lens)

    if no_corruption:
        # 不做 Step1 随机替换，也不做 Step2 mask
        if max_len == args.max_t:
            seq_mask_no_end[:, 0] = False  # 排除 CLS（保持和你训练时 construct_pred 的习惯一致可调整）
            return ids, seq_mask_no_end, torch.zeros_like(ids, dtype=torch.bool)
        else:
            seq_mask = generate_src_mask(max_len, seq_lens + 1)
            return ids, seq_mask_no_end, seq_mask, torch.zeros_like(ids, dtype=torch.bool)

    if max_len == args.max_t:
        seq_mask_no_end[:, 0] = False
        r_indices = torch.randint(0, bitm_model.module.bert.config.vocab_size, ids.shape, device=device)
    else:
        r_indices = torch.randint(0, args.nb_code, ids.shape, device=device)

    corrupt_selector = torch.logical_and(~mask, seq_mask_no_end)
    input_indices = torch.where(corrupt_selector, r_indices, ids)

    # Step 2: Time-step masking
    if probs[0] == 0 and probs[1] == 0:
        mask_token = torch.zeros_like(ids, dtype=torch.bool)
    else:
        # Vectorized probability sampling
        rand_probs = (probs[1] - probs[0]) * torch.rand(batch_size, device=device) + probs[0]
        num_masked = (seq_lens * rand_probs).round().clamp(min=1).long()

        # Selection using topk
        noise = torch.rand((batch_size, max_len), device=device)
        # Force padding/CLS to have lowest priority in selection
        noise.masked_fill_(~seq_mask_no_end, -1.0)

        # Use topk to get indices of tokens to mask
        _, mask_indices = torch.topk(noise, k=num_masked.max().item(), dim=-1)

        # Create mask_token via scatter (faster than full sort)
        mask_token = torch.zeros_like(ids, dtype=torch.bool)
        mask_token.scatter_(1, mask_indices, True)
        # Clean up in case different rows have different num_masked (This handles the clamp/round logic)
        row_idx = torch.arange(max_len, device=device).unsqueeze(0)
        mask_token &= (row_idx < num_masked.unsqueeze(1))

    masked_input_indices = input_indices.masked_fill(mask_token, mask_id)

    if max_len == args.max_t:
        return masked_input_indices, seq_mask_no_end, mask_token
    else:
        seq_mask = generate_src_mask(max_len, seq_lens + 1).to(torch.int64)
        return masked_input_indices, seq_mask_no_end, seq_mask, mask_token

##### ---- Training ---- #####
def get_pred_and_label(pred, ids, seq_mask_no_end):
    # weights[i, j] = 1 / (num_valid * B)
    weights = seq_mask_no_end / (seq_mask_no_end.sum(-1).unsqueeze(-1) * seq_mask_no_end.shape[0])
    pred_seq_masked = pred[seq_mask_no_end, :].view(-1, pred.shape[-1])  # (num_valid, vocab)
    target_seq_masked = ids[seq_mask_no_end]  # (num_valid,)
    weights_seq_masked = weights[seq_mask_no_end]  # (num_valid,)

    return pred_seq_masked, target_seq_masked, weights_seq_masked

def get_loss(pred_masked, target_masked, weights_masked):
    loss = F.cross_entropy(pred_masked, target_masked, reduction='none')
    return (loss * weights_masked).sum()

def split_weighted_ce_loss(pred, target, valid_mask, masked_mask):
    """
    pred: (B, L, V)
    target: (B, L)
    valid_mask: (B, L) bool, 参与loss的token（True才算）
    masked_mask: (B, L) bool, 被mask掉的token（应当只在valid范围内为True）

    返回:
      loss_masked, loss_unmasked, loss_total
    且 loss_masked + loss_unmasked == loss_total
    """
    B, L, V = pred.shape

    # 确保 target dtype 正确
    target = target.long()

    # -------- 1) flatten --------
    pred_flat = pred.reshape(-1, V)  # (B*L, V)
    target_flat = target.reshape(-1)  # (B*L,)
    valid_flat = valid_mask.reshape(-1)  # (B*L,) bool
    masked_flat = masked_mask.reshape(-1)  # (B*L,) bool

    # -------- 2) 只取 valid token --------
    pred_valid = pred_flat[valid_flat]  # (N, V)
    target_valid = target_flat[valid_flat]  # (N,)
    masked_valid = masked_flat[valid_flat]  # (N,) 仅在valid里讨论 masked/unmasked

    # token-wise CE on valid positions only: (N,)
    ce_valid = F.cross_entropy(pred_valid, target_valid, reduction='none')

    # -------- 3) 构造权重：每个 valid token 权重 = 1/(num_valid_i * B) --------
    # 先做一个 (B,L) 的 weights，再 flatten 到 valid
    denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1) * B  # (B,1)
    weights = (valid_mask.float() / denom)  # (B,L)
    w_valid = weights.reshape(-1)[valid_flat]  # (N,)

    # -------- 4) 加权求和，并按 masked/unmasked 拆分 --------
    loss_total = (ce_valid * w_valid).sum()

    loss_masked = (ce_valid[masked_valid] * w_valid[masked_valid]).sum()
    loss_unmasked = (ce_valid[~masked_valid] * w_valid[~masked_valid]).sum()

    return loss_masked, loss_unmasked, loss_total

def train(mask_probs):
    # Get masking probabilities
    probs_m, probs_t = mask_probs[0], mask_probs[1]

    # Get invalid ids for text
    invalid_ids_t = [special_ids_t['eos_id'], special_ids_t['pad_id']]

    ##### ---- Training ---- #####
    best_fid = 1000
    best_iter_m = 0
    best_div = 100
    best_top1 = 0
    best_top2 = 0
    best_top3 = 0
    best_matching = 100

    best_iter_t = 0
    best_bleu1 = 0.
    best_bleu2 = 0.
    best_bleu3 = 0.
    best_bleu4 = 0.
    best_rouge_l = 0.
    best_cider = 0.
    best_bert_f1 = 0.

    for nb_iter in tqdm(range(1, args.total_iter + 1), position=0, leave=True):
        batch = next(train_loader_iter)
        text, token_ids_m, lens_m = batch
        token_ids_m, lens_m = token_ids_m.to(device), lens_m.to(device)
        bs, max_m = token_ids_m.shape[:2]  # (bs, 50)

        mask_id_m = get_model(net).vqvae.num_code + 2
        mask_id_t = tokenizer.mask_token_id  # [MASK] token id in ModernBERT

        # Encode all texts into text tokens for training
        text_inputs = tokenizer(text, padding='max_length', truncation=True, max_length=args.max_t, return_tensors='pt')
        token_ids_t = text_inputs['input_ids'].to(device)  # (bs, max_t)
        seq_mask_t = text_inputs['attention_mask'].to(device)  # (bs, max_t)

        # Get lengths for each text in batch
        valid_mask_t = ~torch.isin(token_ids_t, torch.tensor(invalid_ids_t, device=device))
        lens_t = valid_mask_t.sum(dim=1)  # (bs,)

        # Mask
        masked_input_ids_m, seq_mask_no_end_m, seq_mask_m, mask_token_m = masking(token_ids_m, lens_m, bs, max_m, mask_id_m, probs_m)
        masked_input_ids_t, seq_mask_no_end_t, mask_token_t = masking(token_ids_t, lens_t, bs, args.max_t, mask_id_t, probs_t)
        
        # Train: forward
        logits = bitm_model(masked_input_ids_t, masked_input_ids_m, seq_mask_t, seq_mask_m)

        # Get predictions and targets
        pred_masked_m, target_masked_m, weights_masked_m = get_pred_and_label(logits['logits_m'], token_ids_m, seq_mask_no_end_m)
        pred_masked_t, target_masked_t, weights_masked_t = get_pred_and_label(logits['logits_t'], token_ids_t, seq_mask_no_end_t)

        # Compute loss
        loss_m = get_loss(pred_masked_m, target_masked_m, weights_masked_m)
        loss_t = get_loss(pred_masked_t, target_masked_t, weights_masked_t)
        loss = loss_m + loss_t

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if nb_iter % args.print_iter == 0:
            loss_m_masked, loss_m_unmasked, loss_m = split_weighted_ce_loss(
                pred=logits['logits_m'],
                target=token_ids_m,
                valid_mask=seq_mask_no_end_m,
                masked_mask=mask_token_m
            )
            loss_t_masked, loss_t_unmasked, loss_t = split_weighted_ce_loss(
                pred=logits['logits_t'],
                target=token_ids_t,
                valid_mask=seq_mask_no_end_t,
                masked_mask=mask_token_t
            )

            # [INFO] log loss
            writer.add_scalar('./Loss/motion_masked', loss_m_masked, nb_iter)
            writer.add_scalar('./Loss/motion_unmasked', loss_m_unmasked, nb_iter)
            writer.add_scalar('./Loss/text_masked', loss_t_masked, nb_iter)
            writer.add_scalar('./Loss/text_unmasked', loss_t_unmasked, nb_iter)

            writer.add_scalar('./Loss/motion', loss_m, nb_iter)
            writer.add_scalar('./Loss/text', loss_t, nb_iter)
            writer.add_scalar('./Loss/all', loss, nb_iter)

            # [INFO] log accuracy
            right_masked_m = compute_result(pred_masked_m, token_ids_m, seq_mask_no_end_m)
            right_masked_t = compute_result(pred_masked_t, token_ids_t, seq_mask_no_end_t)
            writer.add_scalar('./ACC/every_motion', right_masked_m * 100 / seq_mask_no_end_m.sum(), nb_iter)
            writer.add_scalar('./ACC/every_text', right_masked_t * 100 / seq_mask_no_end_t.sum(), nb_iter)

            # [INFO] log mask/nomask
            no_mask_token_m = ~mask_token_m * seq_mask_no_end_m
            no_mask_token_t = ~mask_token_t * seq_mask_no_end_t
            writer.add_scalar('./ACC/masked_motion', get_acc(logits['logits_m'], token_ids_m, mask_token_m), nb_iter)
            writer.add_scalar('./ACC/no_masked_motion', get_acc(logits['logits_m'], token_ids_m, no_mask_token_m), nb_iter)
            writer.add_scalar('./ACC/masked_text', get_acc(logits['logits_t'], token_ids_t, mask_token_t), nb_iter)
            writer.add_scalar('./ACC/no_masked_text', get_acc(logits['logits_t'], token_ids_t, no_mask_token_t), nb_iter)

        if nb_iter == 0 or nb_iter % args.eval_iter == 0 or nb_iter == args.total_iter:
            # Test
            if nb_iter == args.total_iter:
                num_repeat = -30
                rand_pos = True
                data_loader = dataset_TM_eval.DATALoaderNew(args.dataname, codebook_test_dir, w_vectorizer, args.nb_code,
                                                           batch_size=32, is_test=True, tokenizer_t=tokenizer,
                                                           max_t=args.max_t, return_all_captions=True)
            # Validation
            else:
                num_repeat = 1
                rand_pos = False
                data_loader = dataset_TM_eval.DATALoaderNew(args.dataname, codebook_val_dir, w_vectorizer, args.nb_code,
                                                           batch_size=32, is_test=False, tokenizer_t=tokenizer,
                                                           max_t=args.max_t, return_all_captions=True)
            # T2M Evaluation
            best_iter_m, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_multi = eval_bitm_t2m(
                args.out_dir, data_loader, net, bitm_model, logger, writer, nb_iter, eval_wrapper, special_ids_m, max_m,
                best_iter=best_iter_m, best_fid=best_fid, best_div=best_div,
                best_top1=best_top1, best_top2=best_top2, best_top3=best_top3, best_matching=best_matching,
                num_repeat=num_repeat, rand_pos=rand_pos)
            # M2T Evaluation
            best_iter_t, best_bleu1, best_bleu2, best_bleu3, best_bleu4, best_rouge_l, best_cider, best_bert_f1 = eval_bitm_m2t(
                args.out_dir, data_loader, bitm_model, logger, writer, nb_iter,
                tokenizer, special_ids_t, invalid_ids_t, max_m, args.max_t,
                best_iter=best_iter_t, best_bleu1=best_bleu1, best_bleu2=best_bleu2, best_bleu3=best_bleu3,
                best_bleu4=best_bleu4, best_rouge_l=best_rouge_l, best_cider=best_cider, best_bert_f1=best_bert_f1,
                num_repeat=num_repeat, rand_pos=rand_pos)

            if nb_iter == args.total_iter:
                msg_final = (f"Train (T2M). Iter {best_iter_m}: FID. {best_fid:.5f}, Diversity. {best_div:.4f}, "
                             f"TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}")
                logger.info(msg_final)
                msg_final = (f"Train (M2T). Iter {best_iter_t}: BLEU1. {best_bleu1:.5f}, BLEU2. {best_bleu2:.4f}, "
                             f"BLEU3. {best_bleu3:.4f}, BLEU4. {best_bleu4:.4f}, ROUGE-L. {best_rouge_l:.4f}, "
                             f"CIDEr. {best_cider:.4f}, BERT-F1. {best_bert_f1:.4f}")
                logger.info(msg_final)
                break

# Training: mix training
# ((prob_lower_bound_m, prob_upper_bound_m), (prob_lower_bound_t, prob_upper_bound_t))
train(mask_probs=((0.5, 1), (0, 0)))
