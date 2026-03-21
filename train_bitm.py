import os
import warnings
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter

from options.option_transformer import get_args_parser
from options.option_rvq import get_opt_parser
from options.get_eval_option import get_opt

from dataset import dataset_tokenize, dataset_TM_train, dataset_TM_eval

from models.evaluator_wrapper import EvaluatorModelWrapper
from models.vqvae import HumanVQVAE
from models.vq.model import RVQVAE
from models.bitm import BiTMBERT

from utils.utils_model import get_logger, initial_optim
from utils.eval_bitm_res import eval_bitm_t2m, eval_bitm_m2t
from exit.utils import get_model, generate_src_mask, init_save_folder, fixseed


warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
device = torch.device('cuda')

##### ---- Exp dirs ---- #####
args = get_args_parser()

init_save_folder(args)

args.vq_dir = f'./output/vq/{args.vq_name}'
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
logger = get_logger(args.out_dir)
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
def momask_opt():
    opt = get_opt_parser()
    fixseed(opt.seed)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/vq/', opt.dataset_name, opt.name)
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == "t2m":
        opt.data_root = './dataset/HumanML3D/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = 263
    elif opt.dataset_name == "kit":
        opt.data_root = './dataset/KIT-ML/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        opt.dim_pose = 251
    else:
        raise KeyError('Dataset Does not Exists')

    return opt

if args.vq_type == 'MMM':
    net = HumanVQVAE(args,  # use args to define different parameters in different quantizers
                     args.nb_code,
                     args.code_dim,
                     args.output_emb_width,
                     args.down_t,
                     args.stride_t,
                     args.width,
                     args.depth,
                     args.dilation_growth_rate)
    resume_pth = f'{args.vq_dir}/net_last.pth'
    print(f'Loading VQ model checkpoint from {resume_pth} ...')
    ckpt = torch.load(resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.to(device)
    net.eval()

    special_ids_m = {
        'end_id': args.nb_code,
        'pad_id': args.nb_code + 1,
        'mask_id': args.nb_code + 2,
    }  # Set motion special ids
    curr_nb_code = args.nb_code

elif args.vq_type == 'MoMask':
    vq_opt = momask_opt()
    net = RVQVAE(vq_opt,
                 vq_opt.dim_pose,
                 vq_opt.nb_code,
                 vq_opt.code_dim,
                 vq_opt.output_emb_width,
                 vq_opt.down_t,
                 vq_opt.stride_t,
                 vq_opt.width,
                 vq_opt.depth,
                 vq_opt.dilation_growth_rate,
                 vq_opt.vq_act,
                 vq_opt.vq_norm)
    resume_pth = pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.vq_name, 'model', 'net_best_fid.tar')
    print(f'Loading VQ model checkpoint from {resume_pth} ...')
    ckpt = torch.load(resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['vq_model' if 'vq_model' in ckpt else 'net'])
    net.to(device)
    net.eval()

    special_ids_m = {
        'end_id': vq_opt.nb_code,
        'pad_id': vq_opt.nb_code + 1,
        'mask_id': vq_opt.nb_code + 2,
    }  # Set motion special ids
    curr_nb_code = vq_opt.nb_code

else:
    raise ValueError(f"Main: the VQ model {args.vq_type} is not supported.")

##### ---- Get Codebook ---- #####
codebooks = {'train': codebook_train_dir, 'val': codebook_val_dir, 'test': codebook_test_dir}
for type, codebook_dir in codebooks.items():
    if len(os.listdir(codebook_dir)) == 0:
        dataloader_token = dataset_tokenize.DATALoaderNew(args.dataname, type, batch_size=1, unit_length=2 ** args.down_t)
        for batch in dataloader_token:
            pose, name = batch
            pose = pose.to(device).float()  # bs, nb_joints, joints_dim, seq_len
            if args.vq_type == 'MMM':
                target = net(pose, type='encode')  # (N, T)
                target = target.cpu().numpy()      # (N, T)
            elif args.vq_type == 'MoMask':
                target, _ = net.encode(pose)           # (N, T, Q)
                target = target[..., 0].cpu().numpy()  # (N, T)
            else:
                raise ValueError(f"Main: the VQ model {args.vq_type} is not supported.")
            np.save(pjoin(codebook_dir, f'{name[0]}.npy'), target)

##### ---- Text2Motion Transformer ---- #####
bitm_model = BiTMBERT(bert_name=bert_name,
                      vq_model=net,
                      vq_type=args.vq_type,
                      special_ids_m=special_ids_m,
                      max_t=args.max_t,
                      max_m=args.max_m,
                      first_modality=args.first_modality,
                      dropout_rate=args.drop_out_rate)

if args.resume_trans is not None:
    print(f'loading transformer checkpoint from {args.resume_trans}')
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    bitm_model.load_state_dict(ckpt['bitm'], strict=True)
bitm_model.to(device)
bitm_model.train()
bitm_model.motion_encoder.vq_model.eval()
bitm_model = torch.nn.DataParallel(bitm_model)

##### ---- Optimizer & Scheduler ---- #####
optimizer = initial_optim(args.decay_option, args.lr, args.weight_decay, bitm_model, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Dataloader ---- #####
train_loader = dataset_TM_train.DATALoaderNew(args.dataname, codebook_train_dir, 'motion_ids', curr_nb_code,
                                              batch_size=args.batch_size, unit_length=2 ** args.down_t)
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
def masking(ids, seq_lens, batch_size, max_len, probs: list = None, no_corruption: bool = False):
    curr_device = ids.device
    is_multi_level = ids.dim() == 3

    if args.pkeep == -1:
        proba = torch.rand(1, device=curr_device)
        mask = torch.rand(ids.shape, device=curr_device) < proba
    else:
        mask = torch.rand(ids.shape, device=curr_device) < args.pkeep

    # Step 1: Random only real token. To prevent pad token got mixed up.
    seq_mask_no_end = generate_src_mask(max_len, seq_lens)
    if max_len == args.max_t:
        seq_mask_no_end[:, 0] = False  # 排除 CLS（保持和你训练时 construct_pred 的习惯一致可调整）
        if no_corruption:
            return ids, seq_mask_no_end, torch.zeros_like(ids, dtype=torch.bool)
        r_indices = torch.randint(0, get_model(bitm_model).bert.config.vocab_size, ids.shape, device=curr_device)
    else:
        if no_corruption:
            return ids, seq_mask_no_end, generate_src_mask(max_len, seq_lens + 1).to(torch.int64), torch.zeros_like(ids, dtype=torch.bool)
        r_indices = torch.randint(0, curr_nb_code, ids.shape, device=curr_device)

    corrupt_selector = torch.logical_and(~mask, seq_mask_no_end.unsqueeze(-1)) if is_multi_level else torch.logical_and(~mask, seq_mask_no_end)
    input_indices = torch.where(corrupt_selector, r_indices, ids)

    # Step 2: Time-step masking
    if probs[0] == 0 and probs[1] == 0:
        mask_token = torch.zeros_like(ids, dtype=torch.bool)
    else:
        # Vectorized probability sampling
        rand_probs = (probs[1] - probs[0]) * torch.rand(batch_size, device=curr_device) + probs[0]
        num_masked = (seq_lens * rand_probs).round().clamp(min=1).long()

        # Selection using topk
        noise = torch.rand((batch_size, max_len), device=curr_device)
        # Force padding/CLS to have lowest priority in selection
        noise.masked_fill_(~seq_mask_no_end, -1.0)

        # Create a 2D masking matrix for implementing scatter
        mask_token_2d = torch.zeros((batch_size, max_len), dtype=torch.bool, device=curr_device)
        # Use topk to get indices of tokens to mask
        _, mask_indices = torch.topk(noise, k=num_masked.max().item(), dim=-1)
        # Generate src for scatter: num_masked --> True, others --> False
        src = torch.arange(mask_indices.shape[1], device=curr_device).unsqueeze(0) < num_masked.unsqueeze(1)
        # Scatter: Invalid k --> False
        mask_token_2d.scatter_(1, mask_indices, src)
        mask_token_2d &= seq_mask_no_end  # 确保只 mask 有效 token

        # Broadcast
        mask_token = mask_token_2d.unsqueeze(-1).expand(-1, -1, ids.shape[-1]) if is_multi_level else mask_token_2d

    if max_len == args.max_t:
        masked_input_indices = input_indices.masked_fill(mask_token, tokenizer.mask_token_id)
        return masked_input_indices, seq_mask_no_end, mask_token
    else:
        masked_input_indices = input_indices.masked_fill(mask_token, special_ids_m['mask_id'])
        return masked_input_indices, seq_mask_no_end, generate_src_mask(max_len, seq_lens + 1).to(torch.int64), mask_token

def task_routing_masking(token_ids_m, lens_m, probs_m: list, token_ids_t, lens_t, probs_t: list, batch_size, task_prob: float):
    # 1. 生成任务分配掩码
    rand_tensor = torch.rand(batch_size, device=token_ids_m.device)
    is_mask_motion = rand_tensor < task_prob  # True: Mask Motion, False: Mask Text
    mask_m_indices = torch.nonzero(is_mask_motion, as_tuple=True)[0]
    mask_t_indices = torch.nonzero(~is_mask_motion, as_tuple=True)[0]
    
    # 2. 得到各任务分别的 batch size
    sub_bs_mask_m = mask_m_indices.shape[0]
    sub_bs_mask_t = mask_t_indices.shape[0]

    # 3. 初始化输出张量
    final_ids_m = token_ids_m.clone()
    final_ids_t = token_ids_t.clone()
    final_mask_token_m = torch.zeros_like(token_ids_m, dtype=torch.bool)
    final_mask_token_t = torch.zeros_like(token_ids_t, dtype=torch.bool)

    # 4. 生成 seq_mask（这是序列长度属性，与路由无关）
    _, seq_mask_no_end_m, seq_mask_m, _ = masking(token_ids_m, lens_m, batch_size, args.max_m, no_corruption=True)
    _, seq_mask_no_end_t, _ = masking(token_ids_t, lens_t, batch_size, args.max_t, no_corruption=True)

    # 5. 仅对需要 Mask Motion 的样本执行 Motion Masking
    if sub_bs_mask_m > 0:
        sub_m_ids = token_ids_m[mask_m_indices]
        sub_m_lens = lens_m[mask_m_indices]
        masked_m, _, _, mask_tok_m = masking(sub_m_ids, sub_m_lens, sub_bs_mask_m, args.max_m, probs_m)
        # 填回原张量
        final_ids_m.index_copy_(0, mask_m_indices, masked_m)
        final_mask_token_m.index_copy_(0, mask_m_indices, mask_tok_m)

    # 6. 仅对需要 Mask Text 的样本执行 Text Masking
    if sub_bs_mask_t > 0:
        sub_t_ids = token_ids_t[mask_t_indices]
        sub_t_lens = lens_t[mask_t_indices]
        masked_t, _, mask_tok_t = masking(sub_t_ids, sub_t_lens, sub_bs_mask_t, args.max_t, probs_t)
        # 填回原张量
        final_ids_t.index_copy_(0, mask_t_indices, masked_t)
        final_mask_token_t.index_copy_(0, mask_t_indices, mask_tok_t)

    return (final_ids_m, seq_mask_no_end_m, seq_mask_m, final_mask_token_m), (final_ids_t, seq_mask_no_end_t, final_mask_token_t)

##### ---- Training ---- #####
def get_pred_and_label(pred, ids, seq_mask_no_end):
    # weights[i, j] = 1 / (num_valid * B)
    weights = seq_mask_no_end / (seq_mask_no_end.sum(-1).unsqueeze(-1) * seq_mask_no_end.shape[0])
    pred_seq_masked = pred[seq_mask_no_end]  # (num_valid, vocab(, Q))
    target_seq_masked = ids[seq_mask_no_end]  # (num_valid(, Q))
    weights_seq_masked = weights[seq_mask_no_end]  # (num_valid(, Q))

    return pred_seq_masked, target_seq_masked, weights_seq_masked

def get_loss(pred_masked, target_masked, weights_masked):
    loss = F.cross_entropy(pred_masked, target_masked, reduction='none')  # (num_valid(, Q))

    if loss.dim() == 2:
        weights_masked = weights_masked.unsqueeze(-1)  # (num_valid,) -> (num_valid, 1)

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

def train(mask_probs, task_prob, split_loss):
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
    best_bleu4 = 0.
    best_rouge_l = 0.
    best_cider = 0.
    best_bert_f1 = 0.

    for nb_iter in tqdm(range(1, args.total_iter + 1), position=0, leave=True):
        batch = next(train_loader_iter)
        # Training Data
        text, token_ids_m, lens_m = batch  # token_ids_m: (batch, T)
        token_ids_m, lens_m = token_ids_m.to(device), lens_m.to(device)
        bs = token_ids_m.shape[0]

        # Encode all texts into text tokens for training
        text_inputs = tokenizer(text, padding='max_length', truncation=True, max_length=args.max_t, return_tensors='pt')
        token_ids_t = text_inputs['input_ids'].to(device)  # (bs, max_t)
        seq_mask_t = text_inputs['attention_mask'].to(device)  # (bs, max_t)

        # Get lengths for each text in batch
        valid_mask_t = ~torch.isin(token_ids_t, torch.tensor(invalid_ids_t, device=device))
        lens_t = valid_mask_t.sum(dim=1)  # (bs,)

        # Mask with task routing
        out_m, out_t = task_routing_masking(token_ids_m, lens_m, probs_m, token_ids_t, lens_t, probs_t, bs, task_prob=task_prob)
        masked_input_ids_m, seq_mask_no_end_m, seq_mask_m, mask_token_m = out_m
        masked_input_ids_t, seq_mask_no_end_t, mask_token_t = out_t
        
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
            if split_loss == True:
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
            if nb_iter == args.total_iter:  # Test
                num_repeat = -30
                rand_pos = True
                codebook_dir = codebook_test_dir
                is_test = True
            else:  # Validation
                num_repeat = 1
                rand_pos = False
                codebook_dir = codebook_val_dir
                is_test = False

            data_loader = dataset_TM_eval.DATALoaderNew(args.dataname, codebook_dir, w_vectorizer, 'motion_ids', curr_nb_code,
                                                        batch_size=32, is_test=is_test, tokenizer_t=tokenizer,
                                                        max_t=args.max_t)
            # T2M Evaluation
            best_iter_m, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_multi = eval_bitm_t2m(
                args.out_dir, data_loader, net, args.vq_type, bitm_model, logger, writer, nb_iter, eval_wrapper, special_ids_m, args.max_m,
                best_iter=best_iter_m, best_fid=best_fid, best_div=best_div,
                best_top1=best_top1, best_top2=best_top2, best_top3=best_top3, best_matching=best_matching,
                num_repeat=num_repeat, rand_pos=rand_pos)
            # M2T Evaluation
            best_iter_t, best_bleu1, best_bleu4, best_rouge_l, best_cider, best_bert_f1 = eval_bitm_m2t(
                args.out_dir, data_loader, bitm_model, logger, writer, nb_iter,
                tokenizer, special_ids_t, invalid_ids_t, args.max_m, args.max_t,
                best_iter=best_iter_t, best_bleu1=best_bleu1, best_bleu4=best_bleu4,
                best_rouge_l=best_rouge_l, best_cider=best_cider, best_bert_f1=best_bert_f1,
                num_repeat=num_repeat, rand_pos=rand_pos)

            if nb_iter == args.total_iter:
                msg_final = (f"Train (T2M). Iter {best_iter_m}: FID. {best_fid:.5f}, Diversity. {best_div:.4f}, "
                             f"TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}")
                logger.info(msg_final)
                msg_final = (f"Train (M2T). Iter {best_iter_t}: BLEU1. {best_bleu1:.5f}, BLEU4. {best_bleu4:.4f}, "
                             f"ROUGE-L. {best_rouge_l:.4f}, CIDEr. {best_cider:.4f}, BERT-F1. {best_bert_f1:.4f}")
                logger.info(msg_final)
                break

# Training
# mask_probs: ((prob_lower_bound_m, prob_upper_bound_m), (prob_lower_bound_t, prob_upper_bound_t))
# task_probs: proportion of T2M samples in a batch
train(mask_probs=((0.5, 1), (0.5, 1)), task_prob=0.8, split_loss=False)
