import os
import warnings
import json
import torch
import torch.nn.functional as F
import numpy as np

import options.option_transformer as option_trans
import models.vqvae as vqvae
import models.t2m_trans as trans
import utils.utils_model as utils_model
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
from dataset import dataset_TM_train, dataset_TM_eval, dataset_tokenize
from exit.utils import get_model, generate_src_mask, init_save_folder
from utils.eval_trans import eval_trans_m, eval_trans_t

from tqdm import tqdm
from einops import rearrange
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

##### ---- ModernBERT ---- #####
model_name = 'answerdotai/modernbert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

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

##### ---- Text2Motion Transformer ---- #####
trans_encoder = trans.Text2Motion_Transformer(vqvae=net,
                                              num_vq=args.nb_code,
                                              embed_dim=args.embed_dim_gpt,
                                              block_size=args.block_size,
                                              num_layers=args.num_layers,
                                              num_local_layer=0,
                                              n_head=args.n_head_gpt,
                                              drop_out_rate=args.drop_out_rate,
                                              fc_rate=args.ff_rate)

if args.resume_trans is not None:
    print('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.to(device)
trans_encoder.train()
trans_encoder = torch.nn.DataParallel(trans_encoder)

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- get codebook ---- #####
codebooks = {'train': codebook_train_dir, 'val': codebook_val_dir, 'test': codebook_test_dir}
for type, codebook_dir in codebooks.items():
    if len(os.listdir(codebook_dir)) == 0:
        dataloader_token = dataset_tokenize.DATALoader(args.dataname, type=type, batch_size=1,
                                                       unit_length=2 ** args.down_t)
        for batch in dataloader_token:
            pose, name = batch
            pose = pose.to(device).float()  # bs, nb_joints, joints_dim, seq_len
            target = net(pose, type='encode')
            target = target.cpu().numpy()
            np.save(os.path.join(codebook_dir, f'{name[0]}.npy'), target)

##### ---- Dataloader ---- #####
train_loader = dataset_TM_train.DATALoader(args.dataname, codebook_train_dir, args.nb_code, args.batch_size,
                                           unit_length=2 ** args.down_t)
train_loader_iter = dataset_TM_train.cycle(train_loader)


##### ---- Training ---- #####
def get_acc(cls_pred, target, mask):
    cls_pred = torch.masked_select(cls_pred, mask.unsqueeze(-1)).view(-1, cls_pred.shape[-1])
    target_all = torch.masked_select(target, mask)
    probs = torch.softmax(cls_pred, dim=-1)
    _, cls_pred_index = torch.max(probs, dim=-1)
    right_num = (cls_pred_index == target_all).sum()
    return right_num * 100 / mask.sum()


def masking(ids, seq_lens: torch.Tensor, batch_size, max_len, mask_id, probs: list):
    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(ids.shape, device=ids.device))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(ids.shape, device=ids.device))

    # Step 1: Random only real token. To prevent pad token got mixed up.
    seq_mask_no_end = generate_src_mask(max_len, seq_lens)
    if max_len == args.max_t:
        seq_mask_no_end[:, 0] = False  # exclude [CLS] token for text
        mask = torch.logical_or(mask, ~seq_mask_no_end).int()
        r_indices = torch.randint_like(ids, trans_encoder.module.bert.vocab_size)
    else:
        mask = torch.logical_or(mask, ~seq_mask_no_end).int()
        r_indices = torch.randint_like(ids, args.nb_code)
    input_indices = mask * ids + (1 - mask) * r_indices

    # Step 2: Time-step masking
    rand_mask_probs = torch.zeros(batch_size, device=seq_lens.device).float().uniform_(probs[0], probs[1])
    num_token_masked = (seq_lens * rand_mask_probs).round().clamp(min=1)
    batch_randperm = torch.rand((batch_size, max_len), device=ids.device) - seq_mask_no_end.int()
    batch_randperm = batch_randperm.argsort(dim=-1)
    mask_token = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
    masked_input_indices = torch.where(mask_token, mask_id, input_indices)

    if max_len == args.max_t:
        return masked_input_indices, seq_mask_no_end, mask_token
    else:
        seq_mask = generate_src_mask(max_len, seq_lens + 1)
        return masked_input_indices, seq_mask_no_end, seq_mask, mask_token


def construct_pred_and_label(pred, ids, seq_mask_no_end):
    weights = seq_mask_no_end / (
                seq_mask_no_end.sum(-1).unsqueeze(-1) * seq_mask_no_end.shape[0])  # weights[i, j] = 1 / (num_valid * B)
    pred_seq_masked = pred[seq_mask_no_end, :].view(-1, pred.shape[-1])  # (num_valid, vocab)
    target_seq_masked = ids[seq_mask_no_end]  # (num_valid,)
    weight_seq_masked = weights[seq_mask_no_end]  # (num_valid,)

    return pred_seq_masked, target_seq_masked, weight_seq_masked


def compute_result(pred_seq_masked, target, seq_mask_no_end):
    probs_seq_masked = torch.softmax(pred_seq_masked, dim=-1)  # (num_valid, vocab)
    _, pred_seq_masked_index = torch.max(probs_seq_masked, dim=-1)  # (num_valid,)
    target_seq_masked = torch.masked_select(target, seq_mask_no_end)  # (num_valid,)
    right_seq_masked = (pred_seq_masked_index == target_seq_masked).sum()  # compare with label

    return right_seq_masked


def train(first_modality, mask_probs):
    # Get masking probabilities
    probs_m, probs_t = mask_probs[0], mask_probs[1]

    # Get special ids for text
    special_ids_t = {
        'mask_id': tokenizer.mask_token_id,
        'cls_id': trans_encoder.module.bert.cls_id,
        'eos_id': trans_encoder.module.bert.eos_id,
        'pad_id': trans_encoder.module.bert.pad_id
    }
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
        masked_input_ids_m, seq_mask_no_end_m, seq_mask_m, mask_token_m = masking(token_ids_m, lens_m, bs, max_m,
                                                                                  mask_id_m, probs_m)
        masked_input_ids_t, seq_mask_no_end_t, mask_token_t = masking(token_ids_t, lens_t, bs, args.max_t, mask_id_t,
                                                                      probs_t)

        # Combine motion mask and text mask
        if first_modality == 'motion':
            src_mask = torch.cat([seq_mask_m, seq_mask_t], dim=1)
        elif first_modality == 'text':
            src_mask = torch.cat([seq_mask_t, seq_mask_m], dim=1)
        else:
            raise RuntimeError(f'The order of the two modalities is not assigned.')

        # Get text embeddings from ModernBERT
        emb_t = trans_encoder.module.bert(input_ids=masked_input_ids_t, attention_mask=seq_mask_t)

        # Train trans: forward
        pred_m, pred_t = trans_encoder(masked_input_ids_m,
                                       src_mask=src_mask,
                                       att_txt=torch.empty(bs, 0, dtype=torch.bool, device=device),
                                       # empty mask for CLS token
                                       word_emb=emb_t,
                                       first=first_modality,
                                       max_m=max_m,
                                       max_t=args.max_t)  # (bs, max_m, vocab), (bs, max_t, vocab)

        # Compute loss as a batch
        pred_seq_masked_m, target_seq_masked_m, weight_seq_masked_m = construct_pred_and_label(pred_m, token_ids_m,
                                                                                               seq_mask_no_end_m)
        pred_seq_masked_t, target_seq_masked_t, weight_seq_masked_t = construct_pred_and_label(pred_t, token_ids_t,
                                                                                               seq_mask_no_end_t)

        loss_m = F.cross_entropy(pred_seq_masked_m, target_seq_masked_m, reduction='none')
        loss_m = (loss_m * weight_seq_masked_m).sum()
        loss_t = F.cross_entropy(pred_seq_masked_t, target_seq_masked_t, reduction='none')
        loss_t = (loss_t * weight_seq_masked_t).sum()
        loss = loss_m + loss_t

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if nb_iter % args.print_iter == 0:
            # [INFO] log loss
            writer.add_scalar('./Loss/motion', loss_m, nb_iter)
            writer.add_scalar('./Loss/text', loss_t, nb_iter)
            writer.add_scalar('./Loss/all', loss, nb_iter)

            # [INFO] log accuracy
            right_seq_masked_m = compute_result(pred_seq_masked_m, token_ids_m, seq_mask_no_end_m)
            right_seq_masked_t = compute_result(pred_seq_masked_t, token_ids_t, seq_mask_no_end_t)
            writer.add_scalar('./ACC/every_motion', right_seq_masked_m * 100 / seq_mask_no_end_m.sum(), nb_iter)
            writer.add_scalar('./ACC/every_text', right_seq_masked_t * 100 / seq_mask_no_end_t.sum(), nb_iter)

            # [INFO] log mask/nomask
            no_mask_token_m = ~mask_token_m * seq_mask_no_end_m
            no_mask_token_t = ~mask_token_t * seq_mask_no_end_t
            writer.add_scalar('./ACC/masked_motion', get_acc(pred_m, token_ids_m, mask_token_m), nb_iter)
            writer.add_scalar('./ACC/no_masked_motion', get_acc(pred_m, token_ids_m, no_mask_token_m), nb_iter)
            writer.add_scalar('./ACC/masked_text', get_acc(pred_t, token_ids_t, mask_token_t), nb_iter)
            writer.add_scalar('./ACC/no_masked_text', get_acc(pred_t, token_ids_t, no_mask_token_t), nb_iter)

        if nb_iter == 0 or nb_iter % args.eval_iter == 0 or nb_iter == args.total_iter:
            if nb_iter == args.total_iter:
                num_repeat = -30
                rand_pos = True
                val_loader = dataset_TM_eval.DATALoaderNew(args.dataname, codebook_test_dir, w_vectorizer, args.nb_code,
                                                           batch_size=32, is_test=True, tokenizer_t=tokenizer,
                                                           max_t=args.max_t)
            else:
                num_repeat = 1
                rand_pos = False
                val_loader = dataset_TM_eval.DATALoaderNew(args.dataname, codebook_val_dir, w_vectorizer, args.nb_code,
                                                           batch_size=32, is_test=False, tokenizer_t=tokenizer,
                                                           max_t=args.max_t)

            best_iter_m, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_multi = eval_trans_m(
                args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter,
                eval_wrapper, max_m, args.max_t, first_modality, best_iter=best_iter_m, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                best_matching=best_matching,
                num_repeat=num_repeat, rand_pos=rand_pos)

            best_iter_t, best_bleu1, best_bleu2, best_bleu3, best_bleu4, best_rouge_l = eval_trans_t(
                args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter,
                eval_wrapper, tokenizer, special_ids_t, invalid_ids_t, max_m, args.max_t, first_modality,
                best_iter=best_iter_t, best_bleu1=best_bleu1, best_bleu2=best_bleu2, best_bleu3=best_bleu3,
                best_bleu4=best_bleu4, best_rouge_l=best_rouge_l,
                num_repeat=num_repeat, rand_pos=rand_pos)

            if nb_iter == args.total_iter:
                msg_final = (f"Train (t2m). Iter {best_iter_m} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, "
                             f"TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}")
                logger.info(msg_final)

                msg_final = (f"Train (m2t). Iter {best_iter_t} : BLEU1. {best_bleu1:.5f}, BLEU2. {best_bleu2:.4f}, "
                             f"BLEU3. {best_bleu3:.4f}, BLEU4. {best_bleu4:.4f}, ROUGE_L. {best_rouge_l:.4f}")
                logger.info(msg_final)
                break


# Training Step 1: mix training
bests = train(
    first_modality='motion',  # "motion" first or "text" first
    mask_probs=((0.5, 1), (0.0001, 1))
    # ((prob_lower_bound_m, prob_upper_bound_m), (prob_lower_bound_t, prob_upper_bound_t))
)
