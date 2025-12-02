import os
import warnings
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import json

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_train, dataset_TM_eval, dataset_tokenize
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper

from tqdm import tqdm
from exit.utils import get_model, generate_src_mask, init_save_folder
from einops import rearrange
import torch.nn.functional as F
from transformers import AutoTokenizer, ModernBertModel

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

init_save_folder(args)

args.vq_dir = f'./output/vq/{args.vq_name}'
codebook_dir = f'{args.vq_dir}/codebook/'
args.resume_pth = f'{args.vq_dir}/net_last.pth'
os.makedirs(args.vq_dir, exist_ok = True)
os.makedirs(codebook_dir, exist_ok = True)
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.out_dir+'/html', exist_ok=True)


##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Network ---- #####
device = torch.device('cuda')
model_name = 'answerdotai/modernbert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
modernbert = ModernBertModel.from_pretrained(model_name).to(device).half()  # float16
modernbert.eval()
for p in modernbert.parameters():
    p.requires_grad = False

class TextModernBERT(torch.nn.Module):
    def __init__(self, model):
        super(TextModernBERT, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 output_hidden_states=False,
                                 return_dict=True)
        return outputs.last_hidden_state.float()  # (bs, max_t, dim)

bert = TextModernBERT(modernbert)
print(f'ModernBERT vocab_size: {bert.model.config.vocab_size}')
print(f'ModernBERT hidden_size: {bert.model.config.hidden_size}')
print(f'ModernBERT mask_token_id: {tokenizer.mask_token_id}')
print(f'ModernBERT pad_token_id: {bert.model.config.pad_token_id}')
print(f'ModernBERT eos_token_id: {bert.model.config.eos_token_id}')
print(f'ModernBERT sep_token_id: {bert.model.config.sep_token_id}')

net = vqvae.HumanVQVAE(args,  # use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)

trans_encoder = trans.Text2Motion_Transformer(vqvae=net,
                                              num_vq=args.nb_code,
                                              num_vt=bert.model.config.vocab_size,
                                              embed_dim=args.embed_dim_gpt,
                                              clip_dim=bert.model.config.hidden_size,
                                              block_size=args.block_size,
                                              num_layers=args.num_layers,
                                              num_local_layer=0,
                                              n_head=args.n_head_gpt,
                                              drop_out_rate=args.drop_out_rate,
                                              fc_rate=args.ff_rate)

print('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

if args.resume_trans is not None:
    print('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()
trans_encoder = torch.nn.DataParallel(trans_encoder)


##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)


##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss(reduction='none')


##### ---- get code ---- #####
##### ---- Dataloader ---- #####
# Encode all motions into motion tokens for training
if len(os.listdir(codebook_dir)) == 0:
    train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2**args.down_t)
    for batch in train_loader_token:
        pose, name = batch
        bs, seq = pose.shape[0], pose.shape[1]

        pose = pose.cuda().float()  # bs, nb_joints, joints_dim, seq_len
        target = net(pose, type='encode')
        target = target.cpu().numpy()
        np.save(pjoin(codebook_dir, name[0] +'.npy'), target)

train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, codebook_dir, unit_length=2**args.down_t)
train_loader_iter = dataset_TM_train.cycle(train_loader)

        
##### ---- Training ---- #####
best_fid = 1000
best_iter = 0
best_div = 100
best_top1 = 0
best_top2 = 0
best_top3 = 0
best_matching = 100

max_t = 77
first_modality = 'motion'  # "motion" or "text"

def get_acc(cls_pred, target, mask):
    cls_pred = torch.masked_select(cls_pred, mask.unsqueeze(-1)).view(-1, cls_pred.shape[-1])
    target_all = torch.masked_select(target, mask)
    probs = torch.softmax(cls_pred, dim=-1)
    _, cls_pred_index = torch.max(probs, dim=-1)
    right_num = (cls_pred_index == target_all).sum()
    return right_num * 100 / mask.sum()

def masking(ids, m_tokens_len, batch_size, max_len, mask_id):
    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(ids.shape, device=ids.device))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(ids.shape, device=ids.device))

    # Step 1: Random only real token. To prevent pad token got mixed up.
    seq_mask_no_end = generate_src_mask(max_len, m_tokens_len)
    if max_len == max_t:
        seq_mask_no_end[:, 0] = False
        mask = torch.logical_or(mask, ~seq_mask_no_end).int()
        r_indices = torch.randint_like(ids, bert.model.config.vocab_size)
    else:
        mask = torch.logical_or(mask, ~seq_mask_no_end).int()
        r_indices = torch.randint_like(ids, args.nb_code)
    input_indices = mask * ids + (1 - mask) * r_indices

    # Step 2: Time-step masking
    rand_mask_probs = torch.zeros(batch_size, device=m_tokens_len.device).float().uniform_(0.5, 1)
    num_token_masked = (m_tokens_len * rand_mask_probs).round().clamp(min=1)
    batch_randperm = torch.rand((batch_size, max_len), device=ids.device) - seq_mask_no_end.int()
    batch_randperm = batch_randperm.argsort(dim=-1)
    mask_token = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
    masked_input_indices = torch.where(mask_token, mask_id, input_indices)

    seq_mask = generate_src_mask(max_len, m_tokens_len + 1)

    return masked_input_indices, seq_mask, seq_mask_no_end, mask_token

def construct_pred_and_label(pred, ids, seq_mask_no_end):
    weights = seq_mask_no_end / (seq_mask_no_end.sum(-1).unsqueeze(-1) * seq_mask_no_end.shape[0])  # weights[i, j] = 1 / (num_valid * B)
    pred_seq_masked = pred[seq_mask_no_end, :].view(-1, pred.shape[-1])  # (num_valid, vocab)
    target_seq_masked = ids[seq_mask_no_end]                             # (num_valid,)
    weight_seq_masked = weights[seq_mask_no_end]                         # (num_valid,)

    return pred_seq_masked, target_seq_masked, weight_seq_masked

def compute_result(pred_seq_masked, target, seq_mask_no_end):
    probs_seq_masked = torch.softmax(pred_seq_masked, dim=-1)         # (num_valid, vocab)
    _, pred_seq_masked_index = torch.max(probs_seq_masked, dim=-1)    # (num_valid,)
    target_seq_masked = torch.masked_select(target, seq_mask_no_end)  # (num_valid,)
    right_seq_masked = (pred_seq_masked_index == target_seq_masked).sum()  # compare with label

    return right_seq_masked

# while nb_iter <= args.total_iter:
for nb_iter in tqdm(range(1, args.total_iter + 1), position=0, leave=True):
    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len = batch
    m_tokens, m_tokens_len = m_tokens.cuda(), m_tokens_len.cuda()
    bs = m_tokens.shape[0]
    target = m_tokens  # (bs, 50)
    target = target.cuda()
    batch_size, max_m = target.shape[:2]

    mask_id = get_model(net).vqvae.num_code + 2
    mask_id_t = tokenizer.mask_token_id  # [MASK] token id in ModernBERT
    
    # Encode all texts into text tokens for training
    inputs = tokenizer(clip_text, padding='max_length', truncation=True, max_length=max_t, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)             # (bs, max_t)
    input_attn_mask = inputs['attention_mask'].to(device)  # (bs, max_t)

    # Get t_tokens_len
    invalid_ids = [bert.model.config.pad_token_id, bert.model.config.eos_token_id, bert.model.config.sep_token_id]
    valid_mask_t = ~torch.isin(input_ids, torch.tensor(invalid_ids, device=input_ids.device))
    t_tokens_len = valid_mask_t.sum(dim=1)

    # Masking
    masked_input_ids, seq_mask, seq_mask_no_end, mask_token = masking(target, m_tokens_len, batch_size, max_m, mask_id)
    masked_input_ids_t, seq_mask_t, seq_mask_no_end_t, mask_token_t = masking(input_ids, t_tokens_len, batch_size, max_t, mask_id_t)
    assert torch.equal(seq_mask_t, input_attn_mask), f'seq_mask_t cannot match the attention mask of text.'

    if first_modality == 'motion':
        src_mask = torch.cat([seq_mask, seq_mask_t], dim=1)
    elif first_modality == 'text':
        src_mask = torch.cat([seq_mask_t, seq_mask], dim=1)
    else:
        raise RuntimeError(f'The order of the two modalities is not assigned.')
    att_txt = torch.empty(batch_size, 0, dtype=torch.bool, device=device)  # empty mask for original MMM's CLS token
    word_emb = bert(input_ids=masked_input_ids_t, attention_mask=seq_mask_t)

    # Training forward
    # output: (bs, max_m, vocab), (bs, max_t, vocab)
    pred_m, pred_t = trans_encoder(
        masked_input_ids, src_mask=src_mask, att_txt=att_txt, word_emb=word_emb, first=first_modality, max_m=max_m, max_t=max_t)

    # Compute xent loss as a batch
    pred_seq_masked, target_seq_masked, weight_seq_masked = construct_pred_and_label(pred_m, target, seq_mask_no_end)
    pred_seq_masked_t, target_seq_masked_t, weight_seq_masked_t = construct_pred_and_label(pred_t, input_ids, seq_mask_no_end_t)

    loss_m = F.cross_entropy(pred_seq_masked, target_seq_masked, reduction='none')
    loss_m = (loss_m * weight_seq_masked).sum()
    loss_t = F.cross_entropy(pred_seq_masked_t, target_seq_masked_t, reduction='none')
    loss_t = (loss_t * weight_seq_masked_t).sum()
    loss = loss_m + loss_t

    ## global loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if nb_iter % args.print_iter == 0:
        right_seq_masked = compute_result(pred_seq_masked, target, seq_mask_no_end)
        right_seq_masked_t = compute_result(pred_seq_masked_t, input_ids, seq_mask_no_end_t)

        writer.add_scalar('./Loss/all', loss, nb_iter)
        writer.add_scalar('./ACC/every_motion', right_seq_masked * 100 / seq_mask_no_end.sum(), nb_iter)
        writer.add_scalar('./ACC/every_text', right_seq_masked_t * 100 / seq_mask_no_end_t.sum(), nb_iter)
        
        # [INFO] log mask/nomask separately
        no_mask_token = ~mask_token * seq_mask_no_end
        no_mask_token_t = ~mask_token_t * seq_mask_no_end_t

        writer.add_scalar('./ACC/masked_motion', get_acc(pred_m, target, mask_token), nb_iter)
        writer.add_scalar('./ACC/no_masked_motion', get_acc(pred_m, target, no_mask_token), nb_iter)
        writer.add_scalar('./ACC/masked_text', get_acc(pred_t, input_ids, mask_token_t), nb_iter)
        writer.add_scalar('./ACC/no_masked_text', get_acc(pred_t, input_ids, no_mask_token_t), nb_iter)

    if nb_iter == 0 or nb_iter % args.eval_iter == 0 or nb_iter == args.total_iter:
        if nb_iter == args.total_iter:
            num_repeat = -3
            rand_pos = True
            val_loader = dataset_TM_eval.DATALoaderNew(args.dataname, True, 32, w_vectorizer,
                                                       tokenizer=tokenizer, max_t=max_t)
        else:
            num_repeat = 1
            rand_pos = False
            val_loader = dataset_TM_eval.DATALoaderNew(args.dataname, False, 32, w_vectorizer,
                                                       tokenizer=tokenizer, max_t=max_t)

        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi = eval_trans.evaluation_transformer_new(
            args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter,
            best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching,
            bert, max_m, max_t, eval_wrapper, first_modality, num_repeat=num_repeat, rand_pos=rand_pos)

    if nb_iter == args.total_iter:
        msg_final = (f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, "
                     f"TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}")
        logger.info(msg_final)
        break
