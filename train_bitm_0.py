import json
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F

import models.vqvae as vqvae
import options.option_transformer as option_trans
import utils.utils_model as utils_model
from dataset import dataset_TM_train, dataset_tokenize
from exit.utils import generate_src_mask, get_model, init_save_folder, maybe_data_parallel
from models.bitm import BiTMBERT
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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

##### ---- VQ-VAE ---- #####
net = vqvae.HumanVQVAE(args,
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


def get_bitm_model(model):
    return get_model(model)


def freeze_motion_stage(model):
    bitm = get_bitm_model(model)
    for param in bitm.parameters():
        param.requires_grad = False

    trainable_modules = [
        bitm.motion_encoder.learn_tok_emb,
        bitm.motion_encoder.proj,
        bitm.motion_encoder.encoder,
        bitm.motion_encoder.norm,
        bitm.motion_decoder,
    ]
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True

    return {
        'frozen': sum(param.numel() for param in bitm.parameters() if not param.requires_grad),
        'trainable': sum(param.numel() for param in bitm.parameters() if param.requires_grad),
    }


def keep_frozen_modules_eval(model):
    bitm = get_bitm_model(model)
    bitm.bert.eval()
    bitm.motion_encoder.vqvae.eval()


##### ---- BiTM ---- #####
bert_name = 'google-bert/bert-large-uncased'
bitm_model = BiTMBERT(bert_name=bert_name,
                      vqvae=net,
                      vocab_m=args.nb_code,
                      max_t=args.max_t,
                      max_m=args.max_m,
                      first_modality=args.first_modality,
                      dropout_rate=args.drop_out_rate,
                      motion_encoder_layers=args.motion_encoder_layers,
                      motion_decoder_layers=args.motion_decoder_layers)

if args.resume_trans is not None:
    print('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    bitm_model.load_state_dict(ckpt['bitm'], strict=True)

bitm_model.to(device)
param_stats = freeze_motion_stage(bitm_model)
bitm_model.train()
keep_frozen_modules_eval(bitm_model)
bitm_model, parallel_info = maybe_data_parallel(
    bitm_model,
    batch_size=args.batch_size,
    min_batch_per_gpu=args.min_batch_per_gpu,
    logger=logger
)
keep_frozen_modules_eval(bitm_model)
logger.info(
    f"Stage0 motion-only training: {param_stats['trainable']:,} params trainable, "
    f"{param_stats['frozen']:,} params frozen."
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
for split, codebook_dir in codebooks.items():
    if len(os.listdir(codebook_dir)) == 0:
        dataloader_token = dataset_tokenize.DATALoaderNew(args.dataname, type=split, batch_size=1,
                                                          unit_length=2 ** args.down_t)
        for batch in dataloader_token:
            pose, name = batch
            pose = pose.to(device).float()
            target = net(pose, type='encode')
            target = target.cpu().numpy()
            np.save(os.path.join(codebook_dir, f'{name[0]}.npy'), target)

##### ---- Dataloader ---- #####
train_loader = dataset_TM_train.DATALoaderNew(args.dataname, codebook_train_dir, args.nb_code, args.batch_size,
                                              unit_length=2 ** args.down_t, split='train', return_caption=False)
train_loader_iter = dataset_TM_train.cycle(train_loader)
val_loader = dataset_TM_train.DATALoaderNew(args.dataname, codebook_val_dir, args.nb_code, args.batch_size,
                                            unit_length=2 ** args.down_t, split='val', return_caption=False,
                                            shuffle=False, drop_last=False)


def masking(ids, seq_lens: torch.Tensor, batch_size, max_len, probs=(0, 1)):
    curr_device = ids.device
    seq_mask_no_end = generate_src_mask(max_len, seq_lens)

    if probs[0] == 0 and probs[1] == 0:
        mask_token = torch.zeros_like(ids, dtype=torch.bool)
    else:
        rand_probs = (probs[1] - probs[0]) * torch.rand(batch_size, 1, device=curr_device) + probs[0]
        mask_token = (torch.rand(ids.shape, device=curr_device) < rand_probs) & seq_mask_no_end

    masked_input_indices = ids.masked_fill(mask_token, special_ids_m['mask_id'])
    seq_mask = generate_src_mask(max_len, seq_lens + 1).to(torch.int64)
    return masked_input_indices, seq_mask_no_end, seq_mask, mask_token


def get_loss(pred, target, loss_mask):
    batch_size, _, vocab_size = pred.shape
    target = target.long()
    flat_mask = loss_mask.reshape(-1)
    if not flat_mask.any():
        return pred.new_tensor(0.0)

    pred_masked = pred.reshape(-1, vocab_size)[flat_mask]
    target_masked = target.reshape(-1)[flat_mask]
    ce_masked = F.cross_entropy(pred_masked, target_masked, reduction='none')

    denom = loss_mask.sum(dim=1, keepdim=True).clamp(min=1) * batch_size
    weights = (loss_mask.float() / denom).reshape(-1)[flat_mask]
    return (ce_masked * weights).sum()


def get_acc(cls_pred, target, mask):
    if mask.sum() == 0:
        return cls_pred.new_tensor(0.0)

    active_outputs = cls_pred[mask]
    active_targets = target[mask]
    predictions = active_outputs.argmax(dim=-1)
    correct = (predictions == active_targets).float().sum()
    return (correct / mask.sum()) * 100


@torch.no_grad()
def evaluate(nb_iter, mask_probs=(0, 1)):
    bitm_model.eval()
    keep_frozen_modules_eval(bitm_model)

    total_loss = 0.0
    total_acc = 0.0
    total_steps = 0

    for batch in tqdm(val_loader, position=1, leave=False):
        token_ids_m, lens_m = batch
        token_ids_m = token_ids_m.to(device)
        lens_m = lens_m.to(device)
        batch_size = token_ids_m.shape[0]

        masked_input_ids_m, _, seq_mask_m, mask_token_m = masking(token_ids_m, lens_m, batch_size, args.max_m,
                                                                  probs=mask_probs)
        if not mask_token_m.any():
            continue

        logits = bitm_model(motion_ids=masked_input_ids_m, motion_mask=seq_mask_m)
        loss = get_loss(logits['logits_m'], token_ids_m, mask_token_m)
        acc = get_acc(logits['logits_m'], token_ids_m, mask_token_m)

        total_loss += loss.item()
        total_acc += acc.item()
        total_steps += 1

    mean_loss = total_loss / total_steps if total_steps > 0 else 0.0
    mean_acc = total_acc / total_steps if total_steps > 0 else 0.0

    writer.add_scalar('./Val/motion_masked_loss', mean_loss, nb_iter)
    writer.add_scalar('./Val/motion_masked_acc', mean_acc, nb_iter)
    logger.info(f"Val. Iter {nb_iter}: motion_masked_loss {mean_loss:.6f}, motion_masked_acc {mean_acc:.4f}")

    bitm_model.train()
    keep_frozen_modules_eval(bitm_model)
    return mean_loss, mean_acc


def save_checkpoint(filename):
    torch.save({'bitm': get_bitm_model(bitm_model).state_dict()}, os.path.join(args.out_dir, filename))


def train(mask_probs=(0, 1)):
    best_val_loss = float('inf')

    for nb_iter in tqdm(range(1, args.total_iter + 1), position=0, leave=True):
        token_ids_m, lens_m = next(train_loader_iter)
        token_ids_m = token_ids_m.to(device)
        lens_m = lens_m.to(device)
        batch_size = token_ids_m.shape[0]

        masked_input_ids_m, _, seq_mask_m, mask_token_m = masking(token_ids_m, lens_m, batch_size, args.max_m,
                                                                  probs=mask_probs)
        logits = bitm_model(motion_ids=masked_input_ids_m, motion_mask=seq_mask_m)
        loss = get_loss(logits['logits_m'], token_ids_m, mask_token_m)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        keep_frozen_modules_eval(bitm_model)

        if nb_iter % args.print_iter == 0:
            masked_acc = get_acc(logits['logits_m'], token_ids_m, mask_token_m)
            writer.add_scalar('./Loss/motion_masked', loss, nb_iter)
            writer.add_scalar('./ACC/motion_masked', masked_acc, nb_iter)
            writer.add_scalar('./Mask/motion_masked_ratio', mask_token_m.float().mean(), nb_iter)
            logger.info(
                f"Train. Iter {nb_iter}: motion_masked_loss {loss.item():.6f}, "
                f"motion_masked_acc {masked_acc.item():.4f}, mask_ratio {mask_token_m.float().mean().item():.4f}"
            )

        if nb_iter % args.eval_iter == 0 or nb_iter == args.total_iter:
            val_loss, _ = evaluate(nb_iter, mask_probs=mask_probs)
            save_checkpoint('net_last.pth')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint('net_best_motion.pth')
                logger.info(f"Best motion checkpoint updated at iter {nb_iter} with val loss {best_val_loss:.6f}.")


train(mask_probs=(0, 1))
