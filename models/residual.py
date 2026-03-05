import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import clip
import time
from einops import repeat
from collections import defaultdict, OrderedDict
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from os.path import join as pjoin

from exit.utils import generate_src_mask, q_schedule, cal_performance, gumbel_sample, top_k, eval_decorator
from utils.losses import print_current_loss
from utils.eval_bitm import eval_res_trans


def def_value():
    return 0.0

class InputProcess(nn.Module):
    def __init__(self, input_feats: int, latent_dim: int):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = x.permute((1, 0, 2)) # [bs, ntokens, input_feats] -> [seqen, bs, input_feats]
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class PositionalEncoding(nn.Module):
    # Borrow from MDM, the same as above, but add dropout, exponential may improve precision
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # [max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) # Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output


class ResidualTransformer(nn.Module):
    def __init__(self, code_dim: int, cond_mode, latent_dim=256, ff_size=1024, num_layers=8, cond_drop_prob=0.1,
                 num_heads=4, dropout=0.1, text_dim=768, shared_codebook=False, share_weight=False,
                 opt=None, **kargs):
        super(ResidualTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.dropout = dropout
        self.opt = opt

        self.cond_mode = cond_mode

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)
        self.cond_drop_prob = cond_drop_prob

        ### Preparing Networks ###
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation='gelu')

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)

        self.encode_quant = partial(F.one_hot, num_classes=self.opt.num_quantizers)
        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        self.quant_emb = nn.Linear(self.opt.num_quantizers, self.latent_dim)
        # if self.cond_mode != 'no_cond':
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.text_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        else:
            raise KeyError("Unsupported condition mode!!!")


        _num_tokens = opt.num_tokens + 1  # one dummy tokens for padding
        self.pad_id = opt.num_tokens

        # self.output_process = OutputProcess_Bert(out_feats=opt.num_tokens, latent_dim=latent_dim)
        self.output_process = OutputProcess(out_feats=code_dim, latent_dim=latent_dim)

        if shared_codebook:
            token_embed = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
            self.token_embed_weight = token_embed.expand(opt.num_quantizers-1, _num_tokens, code_dim)
            if share_weight:
                self.output_proj_weight = self.token_embed_weight
                self.output_proj_bias = None
            else:
                output_proj = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
                output_bias = nn.Parameter(torch.zeros(size=(_num_tokens,)))
                self.output_proj_weight = output_proj.expand(opt.num_quantizers-1, _num_tokens, code_dim)
                self.output_proj_bias = output_bias.expand(opt.num_quantizers-1, _num_tokens)

        else:
            if share_weight:
                self.embed_proj_shared_weight = nn.Parameter(torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 2, _num_tokens, code_dim)))
                self.token_embed_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim)))
                self.output_proj_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim)))
                self.output_proj_bias = None
                self.registered = False
            else:
                output_proj_weight = torch.normal(mean=0, std=0.02,
                                                  size=(opt.num_quantizers - 1, _num_tokens, code_dim))

                self.output_proj_weight = nn.Parameter(output_proj_weight)
                self.output_proj_bias = nn.Parameter(torch.zeros(size=(opt.num_quantizers, _num_tokens)))
                token_embed_weight = torch.normal(mean=0, std=0.02,
                                                  size=(opt.num_quantizers - 1, _num_tokens, code_dim))
                self.token_embed_weight = nn.Parameter(token_embed_weight)

        self.apply(self.__init_weights)
        self.shared_codebook = shared_codebook
        self.share_weight = share_weight

        if self.cond_mode == 'text':
            print(f'Loading text encoder checkpoint from {opt.resume_text_embedding}')
            ckpt = torch.load(opt.resume_text_embedding, map_location='cpu')
            target_weight = ckpt['trans']['bert.embeddings.word_embeddings.weight']
            self.text_encoder = nn.Embedding.from_pretrained(target_weight, freeze=True)  # (vocab_size, hidden_size)
            self.text_encoder.eval()

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_text_encoder(self):
        return [p for name, p in self.named_parameters() if not name.startswith('text_encoder.')]

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def encode_text(self, conds):
        text_embed = self.text_encoder(conds['cond_ids'])  # (batch, max_cond, text_dim)
        text_embed = self.mean_pooling(text_embed, conds['cond_mask'])  # (batch, text_dim)
        return text_embed

    def process_embed_proj_weight(self):
        if self.share_weight and (not self.shared_codebook):
            self.output_proj_weight = torch.cat([self.embed_proj_shared_weight, self.output_proj_weight_], dim=0)
            self.token_embed_weight = torch.cat([self.token_embed_weight_, self.embed_proj_shared_weight], dim=0)

    def output_project(self, logits, qids):
        '''
        :logits: (bs, code_dim, seqlen)
        :qids: (bs)

        :return:
            -logits (bs, ntoken, seqlen)
        '''
        # (num_qlayers-1, num_token, code_dim) -> (bs, ntoken, code_dim)
        output_proj_weight = self.output_proj_weight[qids]
        # (num_qlayers, ntoken) -> (bs, ntoken)
        output_proj_bias = None if self.output_proj_bias is None else self.output_proj_bias[qids]

        output = torch.einsum('bnc, bcs->bns', output_proj_weight, logits)
        if output_proj_bias is not None:
            output += output + output_proj_bias.unsqueeze(-1)
        return output

    def trans_forward(self, motion_codes, qids, cond, padding_mask, force_mask=False):
        '''
        :param motion_codes: (b, seqlen, d)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param qids: (b), quantizer layer ids
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :return:
            -logits: (b, num_token, seqlen)
        '''
        cond = self.mask_cond(cond, force_mask=force_mask)

        # (b, seqlen, d) -> (seqlen, b, latent_dim)
        x = self.input_process(motion_codes)

        # (b, num_quantizer)
        q_onehot = self.encode_quant(qids).float().to(x.device)

        q_emb = self.quant_emb(q_onehot).unsqueeze(0)  # (1, b, latent_dim)
        cond = self.cond_emb(cond).unsqueeze(0)  # (1, b, latent_dim)

        x = self.position_enc(x)
        xseq = torch.cat([cond, q_emb, x], dim=0)  # (seqlen + 2, b, latent_dim)

        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:2]), padding_mask], dim=1)  # (b, seqlen + 2)
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[2:]  # (seqlen, b, e)
        logits = self.output_process(output)
        return logits

    def forward_with_cond_scale(self, motion_codes, q_id, cond_vector, padding_mask, cond_scale=3, force_mask=False):
        bs = motion_codes.shape[0]
        qids = torch.full((bs,), q_id, dtype=torch.long, device=motion_codes.device)
        if force_mask:
            logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, force_mask=True)
            logits = self.output_project(logits, qids-1)
            return logits

        logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask)
        logits = self.output_project(logits, qids-1)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, force_mask=True)
        aux_logits = self.output_project(aux_logits, qids-1)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    def forward(self, all_indices, y, m_lens):
        '''
        :param all_indices: (b, n, q)
        :param y: (b, max_t) ids and (b, max_t) mask for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        '''
        self.process_embed_proj_weight()

        bs, ntokens, num_quant_layers = all_indices.shape
        device = all_indices.device

        # Positions that are PADDED are ALL FALSE
        non_pad_mask = generate_src_mask(ntokens, m_lens)  # (b, n)

        q_non_pad_mask = repeat(non_pad_mask, 'b n -> b n q', q=num_quant_layers)
        all_indices = torch.where(q_non_pad_mask, all_indices, self.pad_id) #(b, n, q)

        # randomly sample quantization layers to work on, [1, num_q)
        active_q_layers = q_schedule(bs, low=1, high=num_quant_layers, device=device)

        # print(self.token_embed_weight.shape, all_indices.shape)
        token_embed = repeat(self.token_embed_weight, 'q c d-> b c d q', b=bs)
        gather_indices = repeat(all_indices[..., :-1], 'b n q -> b n d q', d=token_embed.shape[2])
        # print(token_embed.shape, gather_indices.shape)
        all_codes = token_embed.gather(1, gather_indices)  # (b, n, d, q-1)

        cumsum_codes = torch.cumsum(all_codes, dim=-1) #(b, n, d, q-1)

        active_indices = all_indices[torch.arange(bs), :, active_q_layers]  # (b, n)
        history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers - 1]

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        logits = self.trans_forward(history_sum, active_q_layers, cond_vector, ~non_pad_mask, force_mask)
        logits = self.output_project(logits, active_q_layers-1)
        ce_loss, pred_id, acc = cal_performance(logits, active_indices, ignore_index=self.pad_id)

        return ce_loss, pred_id, acc

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 motion_ids,
                 conds,
                 m_lens,
                 temperature=1,
                 topk_filter_thres=0.9,
                 cond_scale=2,
                 num_res_layers=-1,  # If it's -1, use all.
                 ):
        self.process_embed_proj_weight()

        device = next(self.parameters()).device
        seq_len = motion_ids.shape[1]
        batch_size = len(conds)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~generate_src_mask(seq_len, m_lens)
        motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
        all_indices = [motion_ids]
        history_sum = 0
        num_quant_layers = self.opt.num_quantizers if num_res_layers==-1 else num_res_layers+1

        for i in range(1, num_quant_layers):
            token_embed = self.token_embed_weight[i-1]
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)

            ids = torch.where(padding_mask, self.pad_id, pred_ids)

            motion_ids = ids
            all_indices.append(ids)

        all_indices = torch.stack(all_indices, dim=-1)
        all_indices = torch.where(all_indices==self.pad_id, -1, all_indices)
        return all_indices

    @torch.no_grad()
    @eval_decorator
    def edit(self,
             motion_ids,
             conds,
             m_lens,
             temperature=1,
             topk_filter_thres=0.9,
             cond_scale=2
             ):
        self.process_embed_proj_weight()

        device = next(self.parameters()).device
        seq_len = motion_ids.shape[1]
        batch_size = len(conds)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~generate_src_mask(seq_len, m_lens)
        motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
        all_indices = [motion_ids]
        history_sum = 0

        for i in range(1, self.opt.num_quantizers):
            token_embed = self.token_embed_weight[i-1]
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)

            ids = torch.where(padding_mask, self.pad_id, pred_ids)

            motion_ids = ids
            all_indices.append(ids)

        all_indices = torch.stack(all_indices, dim=-1)
        all_indices = torch.where(all_indices==self.pad_id, -1, all_indices)
        return all_indices


class ResidualTransformerTrainer:
    def __init__(self, args, res_transformer, vq_model, tokenizer, max_cond=77):
        self.opt = args
        self.res_transformer = res_transformer
        self.vq_model = vq_model
        self.tokenizer = tokenizer
        self.max_cond = max_cond
        self.device = args.device
        self.vq_model.eval()

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_res_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def forward(self, batch_data):
        if len(batch_data) == 3:
            conds, motion, m_lens = batch_data
            conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds
            if isinstance(conds[0], str):
                inputs = self.tokenizer(conds, padding='max_length', truncation=True, max_length=self.max_cond, return_tensors='pt')
                conds = {
                    'cond_ids': inputs['input_ids'].to(self.device),  # (bs, max_cond)
                    'cond_mask': inputs['attention_mask'].to(self.device)  # (bs, max_cond)
                }
        elif len(batch_data) == 9:
            _, _, _, _, motion, m_lens, cond_ids, cond_mask, _ = batch_data
            conds = {
                'cond_ids': cond_ids.to(self.device),  # (bs, max_cond)
                'cond_mask': cond_mask.to(self.device)  # (bs, max_cond)
            }
        else:
            raise ValueError("Unexpected batch_data format.")

        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q), (q, b, n ,d)
        code_idx, all_codes = self.vq_model.encode(motion)
        m_lens = m_lens // 4

        ce_loss, pred_ids, acc = self.res_transformer(code_idx, conds, m_lens)

        return ce_loss, acc

    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.opt_res_transformer.zero_grad()
        loss.backward()
        self.opt_res_transformer.step()
        self.scheduler.step()

        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        res_trans_state_dict = self.res_transformer.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            'opt_res_transformer': self.opt_res_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.res_transformer.load_state_dict(checkpoint['res_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_res_transformer.load_state_dict(checkpoint['opt_res_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_wrapper, plot_eval=None):
        self.res_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_res_transformer = optim.AdamW(self.res_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_res_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = eval_res_trans(
            self.opt.save_root, val_loader, self.res_transformer, self.vq_model, self.logger, epoch,
            best_fid=100, best_div=100,
            best_top1=0, best_top2=0, best_top3=0,
            best_matching=100, eval_wrapper=eval_wrapper,
            save_ckpt=False, save_anim=False, plot_func=plot_eval
        )
        best_loss = 100
        best_acc = 0

        while epoch < self.opt.max_epoch:
            self.res_transformer.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)
                logs['loss'] += loss
                logs["acc"] += acc
                logs['lr'] += self.opt_res_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            print('Validation time:')
            self.vq_model.eval()
            self.res_transformer.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            print(f"Validation loss:{np.mean(val_loss):.3f}, Accuracy:{np.mean(val_acc):.3f}")

            self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
            self.logger.add_scalar('Val/acc', np.mean(val_acc), epoch)

            if np.mean(val_loss) < best_loss:
                print(f"Improved loss from {best_loss:.02f} to {np.mean(val_loss)}!!!")
                self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                best_loss = np.mean(val_loss)

            if np.mean(val_acc) > best_acc:
                print(f"Improved acc from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                # self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                best_acc = np.mean(val_acc)

            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = eval_res_trans(
                self.opt.save_root, val_loader, self.res_transformer, self.vq_model, self.logger, epoch, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                best_matching=best_matching, eval_wrapper=eval_wrapper,
                # plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0)
                save_ckpt=True, save_anim=False, plot_func=plot_eval
            )