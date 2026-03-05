import os
import warnings
import json
import torch
from os.path import join as pjoin
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter

from options.option_transformer import get_args_parser
from options.option_rvq import get_opt_parser
from options.get_eval_option import get_opt

from dataset import dataset_TM_eval

from models.evaluator_wrapper import EvaluatorModelWrapper
from models.vqvae import HumanVQVAE
from models.vq.model import RVQVAE
from models.bitm import BiTMBERT

from utils.utils_model import get_logger
from utils.eval_bitm import eval_bitm_t2m, eval_bitm_m2t
from exit.utils import init_save_folder, fixseed


warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
device = torch.device('cuda')

##### ---- Exp dirs ---- #####
args = get_args_parser()

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
        'mask_id': args.nb_code,
        'pad_id': args.nb_code + 1,
    }  # Set motion special ids

else:
    raise ValueError(f"Main: the VQ model {args.vq_type} is not supported.")

##### ---- Text2Motion Transformer ---- #####
bitm_model = BiTMBERT(bert_name=bert_name,
                      vq_model=net,
                      vq_type=args.vq_type,
                      vocab_m=args.nb_code,
                      special_ids_m=special_ids_m,
                      max_t=args.max_t,
                      max_m=args.max_m,
                      first_modality=args.first_modality,
                      dropout_rate=args.drop_out_rate)

if args.resume_trans is not None:
    print('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    bitm_model.load_state_dict(ckpt['trans'], strict=True)
bitm_model.to(device)
bitm_model.eval()
bitm_model = torch.nn.DataParallel(bitm_model)

@torch.no_grad()
def eval_only(split='test'):
    # Get invalid ids for text
    invalid_ids_t = [special_ids_t['eos_id'], special_ids_t['pad_id']]

    # Choose dataset
    if split == 'test':
        num_repeat = -30
        rand_pos = True
        codebook_dir = codebook_test_dir
        is_test = True
    else:
        num_repeat = 1
        rand_pos = False
        codebook_dir = codebook_val_dir
        is_test = False

    data_loader = dataset_TM_eval.DATALoaderNew(args.dataname, codebook_dir, w_vectorizer, 'motion_ids', args.nb_code,
                                                batch_size=32, is_test=is_test, tokenizer_t=tokenizer,
                                                max_t=args.max_t)

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

    nb_iter = 0  # for log

    best_iter_m, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_multi = eval_bitm_t2m(
        args.out_dir, data_loader, net, bitm_model, logger, writer, nb_iter, eval_wrapper, special_ids_m, args.max_m,
        best_iter=best_iter_m, best_fid=best_fid, best_div=best_div,
        best_top1=best_top1, best_top2=best_top2, best_top3=best_top3, best_matching=best_matching,
        num_repeat=num_repeat, rand_pos=rand_pos)

    best_iter_t, best_bleu1, best_bleu4, best_rouge_l, best_cider, best_bert_f1 = eval_bitm_m2t(
        args.out_dir, data_loader, bitm_model, logger, writer, nb_iter,
        tokenizer, special_ids_t, invalid_ids_t, args.max_m, args.max_t,
        best_iter=best_iter_t, best_bleu1=best_bleu1, best_bleu4=best_bleu4,
        best_rouge_l=best_rouge_l, best_cider=best_cider, best_bert_f1=best_bert_f1,
        num_repeat=num_repeat, rand_pos=rand_pos)

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

bests = eval_only(split='test')