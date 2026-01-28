import os
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_train
from dataset import dataset_TM_eval
from dataset import dataset_tokenize
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings

warnings.filterwarnings('ignore')
from exit.utils import get_model, visualize_2motions
from tqdm import tqdm
from exit.utils import get_model, visualize_2motions, generate_src_mask, init_save_folder, uniform, cosine_schedule
from einops import rearrange, repeat
import torch.nn.functional as F
from exit.utils import base_dir

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

# args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
init_save_folder(args)

# [TODO] make the 'output/' folder as arg
args.vq_dir = f'./output/vq/{args.vq_name}'  # os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vq_name}')
codebook_dir = f'{args.vq_dir}/codebook/'
args.resume_pth = f'{args.vq_dir}/net_last.pth'
os.makedirs(args.vq_dir, exist_ok=True)
os.makedirs(codebook_dir, exist_ok=True)
os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(args.out_dir + '/html', exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'),
                                        jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False


# https://github.com/openai/CLIP/issues/111
class TextCLIP(torch.nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        with torch.no_grad():
            word_emb = self.model.token_embedding(text).type(self.model.dtype)
            word_emb = word_emb + self.model.positional_embedding.type(self.model.dtype)
            word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
            word_emb = self.model.transformer(word_emb)
            word_emb = self.model.ln_final(word_emb).permute(1, 0, 2).float()
            enctxt = self.model.encode_text(text).float()
        return enctxt, word_emb


clip_model = TextCLIP(clip_model)

net = vqvae.HumanVQVAE(args,  ## use args to define different parameters in different quantizers
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
                                              embed_dim=args.embed_dim_gpt,
                                              clip_dim=args.clip_dim,
                                              block_size=args.block_size,
                                              num_layers=args.num_layers,
                                              num_local_layer=args.num_local_layer,
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
if len(os.listdir(codebook_dir)) == 0:
    train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2 ** args.down_t)
    for batch in train_loader_token:
        pose, name = batch
        bs, seq = pose.shape[0], pose.shape[1]

        pose = pose.cuda().float()  # bs, nb_joints, joints_dim, seq_len
        target = net(pose, type='encode')
        target = target.cpu().numpy()
        np.save(pjoin(codebook_dir, name[0] + '.npy'), target)

train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, codebook_dir,
                                           unit_length=2 ** args.down_t)
train_loader_iter = dataset_TM_train.cycle(train_loader)

##### ---- Training ---- #####
best_fid = 1000
best_iter = 0
best_div = 100
best_top1 = 0
best_top2 = 0
best_top3 = 0
best_matching = 100


# pred_pose_eval, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, clip_model=clip_model, eval_wrapper=eval_wrapper)

def get_acc(cls_pred, target, mask):
    cls_pred = torch.masked_select(cls_pred, mask.unsqueeze(-1)).view(-1, cls_pred.shape[-1])
    target_all = torch.masked_select(target, mask)
    probs = torch.softmax(cls_pred, dim=-1)
    _, cls_pred_index = torch.max(probs, dim=-1)
    right_num = (cls_pred_index == target_all).sum()
    return right_num * 100 / mask.sum()

@torch.no_grad()
def compute_eval_loss_mmm(val_loader, variant: str, args, net, trans_encoder, clip_model):
    """
    MMM eval loss (motion-only CE), 三个 variant：

    - "nomask": motion不破坏，text正常
    - "mask_text": motion不破坏，text用空串（弱条件）
    - "mask_motion": motion所有“内容token” -> MASK_ID，text正常

    返回：avg_loss（加权 CE，权重策略与训练一致）
    """
    assert variant in ["nomask", "mask_text", "mask_motion"]

    trans_encoder.eval()
    clip_model.eval()
    net.eval()

    mask_id = get_model(net).vqvae.num_code + 2  # 与训练一致：codebook_size + 2

    total_loss = 0.0
    total_weight = 0.0

    for batch in val_loader:
        # val loader __getitem__ 返回：
        # word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token_str, name
        # collate_fn 后通常是 tensor/list 的 batch
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token_str, name = batch

        # caption: list[str]
        if isinstance(caption, tuple):
            caption = list(caption)

        motion = motion.cuda().float()
        m_length = m_length.cuda()

        # 1) 先把连续 motion encode 成训练格式 tokens
        m_tokens, m_tokens_len = encode_val_motion_to_tokens(motion, m_length, net, args)
        bs, max_len = m_tokens.shape

        # 2) 生成 mask（与训练一致）
        seq_mask_no_end = generate_src_mask(max_len, m_tokens_len)        # loss用：只算内容token
        seq_mask = generate_src_mask(max_len, m_tokens_len + 1)           # attention用：包含 END

        # 3) variant：构造 masked 输入
        if variant == "mask_motion":
            masked_input_indices = torch.where(seq_mask_no_end, mask_id, m_tokens)
            caption_in = caption
        elif variant == "mask_text":
            masked_input_indices = m_tokens
            caption_in = [""] * bs
        else:  # "nomask"
            masked_input_indices = m_tokens
            caption_in = caption

        # 4) CLIP text features
        text_tok = clip.tokenize(caption_in, truncate=True).cuda()
        feat_clip_text, word_emb = clip_model(text_tok)

        # 5) forward（注意与训练一致要 [:,1:]）
        logits = trans_encoder(
            masked_input_indices,
            feat_clip_text,
            src_mask=seq_mask,
            att_txt=None,
            word_emb=word_emb
        )[:, 1:]   # (B, max_len, vocab)

        # 6) weighted CE（与训练一致）
        denom = seq_mask_no_end.sum(-1).clamp(min=1).unsqueeze(-1) * bs
        weights = seq_mask_no_end.float() / denom  # (B, L)

        logits_valid = logits[seq_mask_no_end, :].view(-1, logits.shape[-1])  # (N, V)
        target_valid = m_tokens[seq_mask_no_end]                               # (N,)
        w_valid = weights[seq_mask_no_end]                                     # (N,)

        ce = F.cross_entropy(logits_valid, target_valid, reduction='none')     # (N,)
        loss_batch = (ce * w_valid).sum()

        total_loss += loss_batch.item()
        total_weight += w_valid.sum().item()   # 理论上每个 batch ≈ 1

    avg_loss = total_loss / max(total_weight, 1e-9)

    trans_encoder.train()
    return avg_loss

@torch.no_grad()
def encode_val_motion_to_tokens(motion, m_length, net, args):
    """
    motion: (B, T_pad, D) torch.float (val loader输出的 motion)
    m_length: (B,) torch.long/int (真实帧长)
    return:
      m_tokens: (B, max_tok_len) torch.long  [内容tokens + END + PAD]
      m_tokens_len: (B,) torch.long         [内容tokens长度，不含END/PAD]
    """
    device = motion.device
    B = motion.shape[0]

    # 训练集 token 序列的最大长度：和 dataset_TM_train 里的逻辑对齐
    unit_length = 2 ** args.down_t
    max_tok_len = 26 if unit_length == 8 else 50   # 这就是你 training dataset 里那句

    end_id = args.nb_code          # mot_end_idx = codebook_size
    pad_id = args.nb_code + 1      # mot_pad_idx = codebook_size + 1

    # 先全部填 PAD
    m_tokens = torch.full((B, max_tok_len), pad_id, dtype=torch.long, device=device)
    m_tokens_len = torch.zeros((B,), dtype=torch.long, device=device)

    for i in range(B):
        L = int(m_length[i].item())
        # 截掉 padding 的 0 帧，不然 encode 会把 padding 当成真实动作
        x = motion[i, :L]  # (L, D)

        # VQ-VAE encode：不同实现可能吃 (B,T,D) 或 (B,D,T)，这里做个兼容
        x1 = x.unsqueeze(0).float()  # (1, L, D)
        try:
            codes = net(x1, type='encode')
        except Exception:
            # fallback: (1, D, L)
            codes = net(x1.permute(0, 2, 1), type='encode')

        # codes 可能是 (1, T) / (1, T, 1) / (T,) 等形式，统一摊平
        if torch.is_tensor(codes):
            codes = codes.detach()
        codes = codes.reshape(-1).long()  # (T,)

        # 内容 token 长度（要留 1 个位置放 END）
        tok_len = min(int(codes.numel()), max_tok_len - 1)
        tok_len = max(tok_len, 1)  # 防止极端情况下为0导致后面权重除0

        m_tokens_len[i] = tok_len
        m_tokens[i, :tok_len] = codes[:tok_len]
        m_tokens[i, tok_len] = end_id  # append END
        # 后面剩下的已经是 PAD

    return m_tokens, m_tokens_len


# while nb_iter <= args.total_iter:
for nb_iter in tqdm(range(1, args.total_iter + 1), position=0, leave=True):
    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len = batch
    m_tokens, m_tokens_len = m_tokens.cuda(), m_tokens_len.cuda()
    bs = m_tokens.shape[0]
    target = m_tokens  # (bs, 26)
    target = target.cuda()
    batch_size, max_len = target.shape[:2]

    # Random Drop Text
    # text_mask = np.random.random(len(clip_text)) > .05
    # clip_text = np.array(clip_text)
    # clip_text[~text_mask] = ''

    text = clip.tokenize(clip_text, truncate=True).cuda()

    feat_clip_text, word_emb = clip_model(text)

    # [INFO] Swap input tokens
    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(target.shape,
                                                  device=target.device))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(target.shape,
                                                       device=target.device))
    # random only motion token (not pad token). To prevent pad token got mixed up.
    seq_mask_no_end = generate_src_mask(max_len, m_tokens_len)
    mask = torch.logical_or(mask, ~seq_mask_no_end).int()
    r_indices = torch.randint_like(target, args.nb_code)
    input_indices = mask * target + (1 - mask) * r_indices

    # Time step masking
    mask_id = get_model(net).vqvae.num_code + 2
    # rand_time = uniform((batch_size,), device = target.device)
    # rand_mask_probs = cosine_schedule(rand_time)
    rand_mask_probs = torch.zeros(batch_size, device=m_tokens_len.device).float().uniform_(0.5, 1)
    # rand_mask_probs = cosine_schedule(rand_mask_probs)
    num_token_masked = (m_tokens_len * rand_mask_probs).round().clamp(min=1)
    seq_mask = generate_src_mask(max_len, m_tokens_len + 1)
    batch_randperm = torch.rand((batch_size, max_len), device=target.device) - seq_mask_no_end.int()
    batch_randperm = batch_randperm.argsort(dim=-1)
    mask_token = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

    # masked_target = torch.where(mask_token, input=input_indices, other=-1)
    masked_input_indices = torch.where(mask_token, mask_id, input_indices)

    att_txt = None  # CFG: torch.rand((seq_mask.shape[0], 1)) > 0.1
    cls_pred = \
    trans_encoder(masked_input_indices, feat_clip_text, src_mask=seq_mask, att_txt=att_txt, word_emb=word_emb)[:, 1:]

    # [INFO] Compute xent loss as a batch
    weights = seq_mask_no_end / (seq_mask_no_end.sum(-1).unsqueeze(-1) * seq_mask_no_end.shape[0])
    cls_pred_seq_masked = cls_pred[seq_mask_no_end, :].view(-1, cls_pred.shape[-1])
    target_seq_masked = target[seq_mask_no_end]
    weight_seq_masked = weights[seq_mask_no_end]
    loss_cls = F.cross_entropy(cls_pred_seq_masked, target_seq_masked, reduction='none')
    loss_cls = (loss_cls * weight_seq_masked).sum()

    ## global loss
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    if nb_iter % args.print_iter == 0:
        probs_seq_masked = torch.softmax(cls_pred_seq_masked, dim=-1)
        _, cls_pred_seq_masked_index = torch.max(probs_seq_masked, dim=-1)
        target_seq_masked = torch.masked_select(target, seq_mask_no_end)
        right_seq_masked = (cls_pred_seq_masked_index == target_seq_masked).sum()

        writer.add_scalar('./Loss/all', loss_cls, nb_iter)
        writer.add_scalar('./ACC/every_token', right_seq_masked * 100 / seq_mask_no_end.sum(), nb_iter)

        # [INFO] log mask/nomask separately
        no_mask_token = ~mask_token * seq_mask_no_end
        writer.add_scalar('./ACC/masked', get_acc(cls_pred, target, mask_token), nb_iter)
        writer.add_scalar('./ACC/no_masked', get_acc(cls_pred, target, no_mask_token), nb_iter)

        # msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
        # logger.info(msg)

    if nb_iter == 0 or nb_iter % args.eval_iter == 0 or nb_iter == args.total_iter:
        num_repeat = 1
        rand_pos = False
        if nb_iter == args.total_iter:
            num_repeat = -30
            rand_pos = True
            val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)

        # === 额外计算并记录 eval loss（nomask / mask_text / mask_motion）===
        for variant in ["nomask", "mask_text", "mask_motion"]:
            eval_loss = compute_eval_loss_mmm(val_loader, variant, args, net, trans_encoder, clip_model)
            writer.add_scalar(f'./Eval_Loss/{variant}/motion', eval_loss, nb_iter)
            logger.info(f"EvalLoss[{variant}] (iter {nb_iter}): {eval_loss:.4f}")

        pred_pose_eval, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_transformer_old(
            args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter, best_fid, best_iter, best_div,
            best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper,
            dataname=args.dataname, num_repeat=num_repeat, rand_pos=rand_pos)
        # for i in range(4):
        #     x = pose[i].detach().cpu().numpy()
        #     y = pred_pose_eval[i].detach().cpu().numpy()
        #     l = m_length[i]
        #     caption = clip_text[i]
        #     cleaned_name = '-'.join(caption[:200].split('/'))
        #     visualize_2motions(x, val_loader.dataset.std, val_loader.dataset.mean, args.dataname, l, y, save_path=f'{args.out_dir}/html/{str(nb_iter)}_{cleaned_name}_{l}.html')

    if nb_iter == args.total_iter:
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break
