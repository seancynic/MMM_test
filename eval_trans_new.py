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
trans_encoder = torch.nn.DataParallel(trans_encoder)

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

@torch.no_grad()
def eval_only(first_modality, split="test"):
    # ===== special ids for text =====
    special_ids_t = {
        'mask_id': tokenizer.mask_token_id,
        'cls_id': trans_encoder.module.bert.cls_id if isinstance(trans_encoder, torch.nn.DataParallel) else trans_encoder.bert.cls_id,
        'eos_id': trans_encoder.module.bert.eos_id if isinstance(trans_encoder, torch.nn.DataParallel) else trans_encoder.bert.eos_id,
        'pad_id': trans_encoder.module.bert.pad_id if isinstance(trans_encoder, torch.nn.DataParallel) else trans_encoder.bert.pad_id,
    }
    invalid_ids_t = [special_ids_t['eos_id'], special_ids_t['pad_id']]

    # ===== 选择评估集 =====
    if split == "test":
        num_repeat = -30
        rand_pos = True
        codebook_dir = codebook_test_dir
        is_test = True
    else:
        num_repeat = 1
        rand_pos = False
        codebook_dir = codebook_val_dir
        is_test = False

    # ===== dataloader =====
    val_loader = dataset_TM_eval.DATALoaderNew(
        args.dataname, codebook_dir, w_vectorizer, args.nb_code,
        batch_size=32, is_test=is_test, tokenizer_t=tokenizer, max_t=args.max_t
    )

    # ===== 切到 eval 模式 =====
    net.eval()
    trans_encoder.eval()

    # max_m：eval_trans_m / eval_trans_t 要用到 max_m
    # 你原来是从 train batch 里拿 max_m，这里我们直接用一个常见默认（通常是 50）
    # 如果你 args 里有 max_m 或 block_size 对应的 motion 长度，也可以改成 args.max_m
    max_m = 50

    # ===== 初始化 best 值（eval_only不需要“best”，但保持函数签名兼容）=====
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

    nb_iter = 0  # 只是日志用

    # ===== motion 侧评估 =====
    best_iter_m, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_multi = eval_trans_m(
        args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter,
        eval_wrapper, max_m, args.max_t, first_modality,
        best_iter=best_iter_m, best_fid=best_fid,
        best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
        best_matching=best_matching,
        num_repeat=num_repeat, rand_pos=rand_pos
    )

    # ===== text 侧评估 =====
    best_iter_t, best_bleu1, best_bleu2, best_bleu3, best_bleu4, best_rouge_l, best_cider, best_bert_f1 = eval_trans_t(
        args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter,
        eval_wrapper, tokenizer, special_ids_t, invalid_ids_t,
        max_m, args.max_t, first_modality,
        best_iter=best_iter_t,
        best_bleu1=best_bleu1, best_bleu2=best_bleu2, best_bleu3=best_bleu3, best_bleu4=best_bleu4,
        best_rouge_l=best_rouge_l,
        best_cider=best_cider, best_bert_f1=best_bert_f1,
        num_repeat=num_repeat, rand_pos=rand_pos
    )

    # ===== 打印汇总 =====
    logger.info(
        f"[EVAL {split}] (t2m) FID {best_fid:.5f}, Div {best_div:.4f}, "
        f"TOP1 {best_top1:.4f}, TOP2 {best_top2:.4f}, TOP3 {best_top3:.4f}, Match {best_matching:.4f}"
    )
    logger.info(
        f"[EVAL {split}] (m2t) BLEU1 {best_bleu1:.5f}, BLEU2 {best_bleu2:.4f}, BLEU3 {best_bleu3:.4f}, "
        f"BLEU4 {best_bleu4:.4f}, ROUGE_L {best_rouge_l:.4f}, CIDEr {best_cider:.4f}, BERT_F1 {best_bert_f1:.4f}"
    )

    return {
        "fid": float(best_fid),
        "div": float(best_div),
        "top1": float(best_top1),
        "top2": float(best_top2),
        "top3": float(best_top3),
        "matching": float(best_matching),
        "bleu1": float(best_bleu1),
        "bleu2": float(best_bleu2),
        "bleu3": float(best_bleu3),
        "bleu4": float(best_bleu4),
        "rouge_l": float(best_rouge_l),
        "cider": float(best_cider),
        "bert_f1": float(best_bert_f1),
    }

# Training Step 1: mix training
bests = eval_only(
    first_modality='motion',  # "motion" first or "text" first
    split='test',
)
