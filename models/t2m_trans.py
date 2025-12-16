import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import models.pos_encoding as pos_encoding
from exit.utils import cosine_schedule, uniform, top_k, gumbel_sample, top_p
from einops import rearrange
from exit.utils import get_model, generate_src_mask
from transformers import ModernBertModel


class PatchUpSampling(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.up_sampling = nn.Linear(dim, 4 * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, F, C
        """
        x = self.norm(x)
        x = self.up_sampling(x)
        x0 = x[:, :, 0::4]  
        x1 = x[:, :, 1::4]
        x2 = x[:, :, 2::4]
        x3 = x[:, :, 3::4]
        x = torch.cat([x0, x1, x2, x3], 1)  
        return x

class Decoder_Transformer(nn.Module):
    def __init__(self, 
                code_dim=1024, 
                embed_dim=512, 
                output_dim=263,
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        
        super().__init__()
        self.joint_embed = nn.Linear(code_dim, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.up_sample = PatchUpSampling(embed_dim)
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim),
                            nn.Linear(embed_dim, output_dim))
        self.block_size = block_size
        self.n_head = n_head
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, token_embeddings):
        # token_embeddings = self.tok_emb(idx)
        # B, T = src_mask.shape
        # src_mask = src_mask.view(B, 1, 1, T).repeat(1, self.n_head, T, 1)

        token_embeddings = token_embeddings.permute(0, 2, 1)
        token_embeddings = self.joint_embed(token_embeddings)
        x = self.pos_embed(token_embeddings)


        for block in self.blocks:
            x = block(x)
        x = self.up_sample(x)


        x = self.head(x).permute(0, 2, 1)
        return x

# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L342C9-L343C33
class PatchMerging(nn.Module):
    def __init__(self, input_feats, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * input_feats, dim, bias=False)
        self.norm = norm_layer(4 * input_feats)

    def forward(self, x):
        """
        x: B, F, C
        """
        x0 = x[:, 0::4, :]  # B F/2 C
        x1 = x[:, 1::4, :]
        x2 = x[:, 2::4, :]  # B F/2 C
        x3 = x[:, 3::4, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B F/2 2*C
        x = self.norm(x)
        x = self.reduction(x)
        return x

class Encoder_Transformer(nn.Module):
    def __init__(self, 
                input_feats=1024, 
                embed_dim=512, 
                output_dim=263,
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        
        super().__init__()
        self.joint_embed = nn.Linear(input_feats, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.weighted_mean_norm = nn.LayerNorm(embed_dim)
        self.weighted_mean = torch.nn.Conv1d(in_channels=block_size, out_channels=1, kernel_size=1)

        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim),
                            nn.Linear(embed_dim, output_dim))
        self.block_size = block_size
        self.n_head = n_head
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, joints):
        # B, T = src_mask.shape

        joints = joints.permute(0,2,1)
        # token_embeddings = self.joint_embed(joints)

        block_step_len = int(len(self.blocks)/3)

        x = self.joint_embed(joints)
        token_len = int(x.shape[1]/self.block_size)
        _original_shape = list(x.shape)
        x = x.view(x.shape[0]*token_len, self.block_size, -1)

        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.weighted_mean_norm(x)
        x = self.weighted_mean(x)
        _original_shape[1] = int(_original_shape[1] / self.block_size)
        x = x.view(*_original_shape)

        x = self.head(x).permute(0, 2, 1)
        return x

class TextModernBERT(torch.nn.Module):
    def __init__(self, model):
        super(TextModernBERT, self).__init__()
        self.model = model
        self.vocab_size = model.config.vocab_size
        self.cls_id = model.config.cls_token_id
        self.eos_id = model.config.eos_token_id
        self.pad_id = model.config.pad_token_id

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 output_hidden_states=False,
                                 return_dict=True)
        return outputs.last_hidden_state.float()  # (bs, max_t, dim)

class Text2Motion_Transformer(nn.Module):

    def __init__(self, 
                vqvae,
                num_vq=1024,
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                num_local_layer=0, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        # ModernBERT
        model_name = 'answerdotai/modernbert-base'
        modernbert = ModernBertModel.from_pretrained(model_name).half()  # float16
        modernbert.eval()
        for p in modernbert.parameters():
            p.requires_grad = False
        self.bert = TextModernBERT(modernbert)

        self.n_head = n_head
        self.trans_base = CrossCondTransBase(vqvae, num_vq, embed_dim, modernbert.config.hidden_size, block_size, num_layers, num_local_layer, n_head, drop_out_rate, fc_rate)
        self.trans_head = CrossCondTransHead(num_vq, modernbert.config.vocab_size, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_vq = num_vq
        # self.skip_trans = Skip_Connection_Transformer(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)

    def get_block_size(self):
        return self.block_size

    def forward(self, *args, type='forward', **kwargs):
        '''type=[forward, sample]'''
        if type=='forward':
            return self.forward_function(*args, **kwargs)
        elif type=='sample':
            return self.sample(*args, **kwargs)
        elif type=='sample_m':
            return self.sample_m(*args, **kwargs)
        elif type=='sample_t':
            return self.sample_t(*args, **kwargs)
        elif type=='inpaint':
            return self.inpaint(*args, **kwargs)
        else:
            raise ValueError(f'Unknown "{type}" type')
        
    def get_attn_mask(self, src_mask, att_txt=None):
        if att_txt is None:
            att_txt = torch.tensor([[True]]*src_mask.shape[0]).to(src_mask.device)
        src_mask = torch.cat([att_txt, src_mask],  dim=1)
        B, T = src_mask.shape
        src_mask = src_mask.view(B, 1, 1, T).repeat(1, self.n_head, T, 1)
        return src_mask

    def forward_function(self, idxs, clip_feature=None, src_mask=None, att_txt=None, word_emb=None, first=None, max_m=None, max_t=None):
        if src_mask is not None:
            src_mask = self.get_attn_mask(src_mask, att_txt)
            src_mask = src_mask.bool()  # int64 to bool
        feat = self.trans_base(idxs, clip_feature, src_mask, word_emb, first)
        logits = self.trans_head(feat, src_mask, first, max_m, max_t)

        return logits

    def sample(self, clip_feature, word_emb, m_length=None, if_test=False, rand_pos=True, CFG=-1, token_cond=None, max_steps = 10):
        max_length = 49
        batch_size = clip_feature.shape[0]
        mask_id = self.num_vq + 2
        pad_id = self.num_vq + 1
        end_id = self.num_vq
        shape = (batch_size, self.block_size - 1)
        topk_filter_thres = .9
        starting_temperature = 1.0
        scores = torch.ones(shape, dtype = torch.float32, device = clip_feature.device)
        
        m_tokens_len = torch.ceil((m_length)/4).long()
        src_token_mask = generate_src_mask(self.block_size-1, m_tokens_len+1)
        src_token_mask_noend = generate_src_mask(self.block_size-1, m_tokens_len)
        if token_cond is not None:
            ids = token_cond.clone()
            ids[~src_token_mask_noend] = pad_id
            num_token_cond = (ids==mask_id).sum(-1)
        else:
            ids = torch.full(shape, mask_id, dtype = torch.long, device = clip_feature.device)
        
        # [TODO] confirm that these 2 lines are not neccessary (repeated below and maybe don't need them at all)
        ids[~src_token_mask] = pad_id # [INFO] replace with pad id
        ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # [INFO] replace with end id

        sample_max_steps = torch.round(max_steps/max_length*m_tokens_len) + 1e-8
        for step in range(max_steps):
            timestep = torch.clip(step/(sample_max_steps), max=1)
            if len(m_tokens_len)==1 and step > 0 and torch.clip(step-1/(sample_max_steps), max=1).cpu().item() == timestep:
                break
            rand_mask_prob = cosine_schedule(timestep) # timestep #
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)

            if token_cond is not None:
                num_token_masked = (rand_mask_prob * num_token_cond).long().clip(min=1)
                scores[token_cond!=mask_id] = 0
            
            # [INFO] rm no motion frames
            scores[~src_token_mask_noend] = 0
            scores = scores/scores.sum(-1)[:, None] # normalize only unmasked token
            
            # if rand_pos:
            #     sorted_score_indices = scores.multinomial(scores.shape[-1], replacement=False) # stocastic
            # else:
            sorted, sorted_score_indices = scores.sort(descending=True) # deterministic
            
            ids[~src_token_mask] = pad_id # [INFO] replace with pad id
            ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # [INFO] replace with end id
            ## [INFO] Replace "mask_id" to "ids" that have highest "num_token_masked" "scores" 
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
            # [INFO] repeat last_id to make it scatter_ the existing last ids.
            last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1)-1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index*~select_masked_indices)
            ids.scatter_(-1, sorted_score_indices, mask_id)

            logits = self.forward(ids, clip_feature, src_token_mask, word_emb=word_emb)[:,1:]
            filtered_logits = logits #top_p(logits, .5) # #top_k(logits, topk_filter_thres)
            if rand_pos:
                temperature = 1 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed
            else:
                temperature = 0 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            # [INFO] if temperature==0: is equal to argmax (filtered_logits.argmax(dim = -1))
            # pred_ids = filtered_logits.argmax(dim = -1)
            pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            is_mask = ids == mask_id

            ids = torch.where(is_mask, pred_ids, ids)
            
            # if timestep == 1.:
            #     print(probs_without_temperature.shape)
            probs_without_temperature = logits.softmax(dim = -1)
            scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')
            scores = scores.masked_fill(~is_mask, 0)
        if if_test:
            return ids
        return ids

    def sample_m(self, lens_m=None, word_emb=None, seq_mask_t=None,
                 max_m=50, max_t=77, rand_pos=True, token_cond=None, max_steps=10, first=None):
        max_length = max_m - 1  # 49
        batch_size = lens_m.shape[0]

        # set special ids
        mask_id = self.num_vq + 2
        pad_id = self.num_vq + 1
        end_id = self.num_vq

        # generate target token masks
        shape = (batch_size, max_m)
        seq_mask_m = generate_src_mask(max_m, lens_m + 1)  # target token: motion
        seq_mask_no_end_m = generate_src_mask(max_m, lens_m)  # target token: motion

        # init sampling score
        scores = torch.ones(shape, dtype=torch.float32, device=lens_m.device)

        # init motion token ids
        if token_cond is not None:  # has partial condition
            token_ids_m = token_cond.clone()
            token_ids_m[~seq_mask_no_end_m] = pad_id
            num_token_cond = (token_ids_m == mask_id).sum(-1)
        else:  # start from full mask
            token_ids_m = torch.full(shape, mask_id, dtype=torch.long, device=lens_m.device)

        ## TODO: confirm that these 2 lines are not necessary (repeated below and maybe don't need them at all)
        token_ids_m[~seq_mask_m] = pad_id                           # replace with pad id
        token_ids_m.scatter_(-1, lens_m[..., None].long(), end_id)  # replace with end id

        sample_max_steps = torch.round(max_steps / max_length * lens_m) + 1e-8
        for step in range(max_steps):
            timestep = torch.clip(step / sample_max_steps, max=1)
            if len(lens_m) == 1 and step > 0 and torch.clip(step - 1 / sample_max_steps, max=1).cpu().item() == timestep:
                break
            rand_mask_prob = cosine_schedule(timestep)
            num_token_masked = (rand_mask_prob * lens_m).long().clip(min=1)

            if token_cond is not None:
                num_token_masked = (rand_mask_prob * num_token_cond).long().clip(min=1)
                scores[token_cond != mask_id] = 0

            # remove no motion frames
            scores[~seq_mask_no_end_m] = 0
            scores = scores / scores.sum(-1)[:, None]  # normalize only unmasked token

            _, sorted_score_indices = scores.sort(descending=True)  # deterministic

            token_ids_m[~seq_mask_m] = pad_id                           # replace with pad id
            token_ids_m.scatter_(-1, lens_m[..., None].long(), end_id)  # replace with end id

            # replace "mask_id" to "ids" that have highest "num_token_masked" "scores"
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
            # repeat last_id to make it scatter_ the existing last ids
            last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1) - 1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index * ~select_masked_indices)
            token_ids_m.scatter_(-1, sorted_score_indices, mask_id)

            if first == 'motion':
                src_mask = torch.cat([seq_mask_m, seq_mask_t], dim=1)
            elif first == 'text':
                src_mask = torch.cat([seq_mask_t, seq_mask_m], dim=1)
            else:
                raise RuntimeError(f'The order of the two modalities is not assigned.')

            pred_m, _ = self.forward(token_ids_m,
                                     src_mask=src_mask,
                                     att_txt=torch.empty(batch_size, 0, dtype=torch.bool, device=lens_m.device),  # empty mask for CLS token
                                     word_emb=word_emb,
                                     first=first,
                                     max_m=max_m,
                                     max_t=max_t)

            if rand_pos:
                temperature = 1  # starting_temperature * (steps_until_x0 / timesteps)  # temperature is annealed
            else:
                temperature = 0  # starting_temperature * (steps_until_x0 / timesteps)  # temperature is annealed

            # if temperature == 0, it is equal to argmax (pred_ids = pred_m.argmax(dim=-1))
            pred_ids_m = gumbel_sample(pred_m, temperature=temperature, dim=-1)
            is_mask = token_ids_m == mask_id

            token_ids_m = torch.where(is_mask, pred_ids_m, token_ids_m)

        return token_ids_m

    def sample_t(self, special_ids=None, lens_t=None, token_ids_m=None, seq_mask_m=None,
                 max_m=50, max_t=77, rand_pos=True, token_cond=None, max_steps=10, first=None):
        batch_size = lens_t.shape[0]

        # generate target token masks
        shape = (batch_size, max_t)
        seq_mask_t = generate_src_mask(max_t, lens_t + 1)  # target token: text
        seq_mask_no_end_t = generate_src_mask(max_t, lens_t)  # target token: text

        scores = torch.ones(shape, dtype=torch.float32, device=lens_t.device)

        # init text token ids
        if token_cond is not None:  # has partial condition
            token_ids_t = token_cond.clone()
            token_ids_t[~seq_mask_no_end_t] = special_ids['pad_id']
            num_token_cond = (token_ids_t == special_ids['mask_id']).sum(-1)
        else:  # start from full mask
            token_ids_t = torch.full(shape, special_ids['mask_id'], dtype=torch.long, device=lens_t.device)
            token_ids_t[:, 0] = special_ids['cls_id']  # add [CLS] token for text

        ## TODO: confirm that these 2 lines are not necessary (repeated below and maybe don't need them at all)
        token_ids_t[~seq_mask_t] = special_ids['pad_id']                           # replace with pad id
        token_ids_t.scatter_(-1, lens_t[..., None].long(), special_ids['eos_id'])  # replace with end id

        sample_max_steps = torch.round(max_steps / max_t * lens_t) + 1e-8
        for step in range(max_steps):
            timestep = torch.clip(step / sample_max_steps, max=1)
            if len(lens_t) == 1 and step > 0 and torch.clip(step - 1 / sample_max_steps, max=1).cpu().item() == timestep:
                break
            rand_mask_prob = cosine_schedule(timestep)
            num_token_masked = (rand_mask_prob * lens_t).long().clip(min=1)

            if token_cond is not None:
                num_token_masked = (rand_mask_prob * num_token_cond).long().clip(min=1)
                scores[token_cond != special_ids['mask_id']] = 0

            # Set sampling score to 0 for [PAD] and [CLS]
            scores[~seq_mask_no_end_t] = 0
            scores[:, 0] = 0
            scores = scores / scores.sum(-1)[:, None]  # normalize only unmasked token

            _, sorted_score_indices = scores.sort(descending=True)  # deterministic

            token_ids_t[~seq_mask_t] = special_ids['pad_id']                           # replace with pad id
            token_ids_t.scatter_(-1, lens_t[..., None].long(), special_ids['eos_id'])  # replace with end id

            # replace "mask_id" to "ids" that have highest "num_token_masked" "scores"
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
            # repeat last_id to make it scatter_ the existing last ids
            last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1) - 1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index * ~select_masked_indices)
            token_ids_t.scatter_(-1, sorted_score_indices, special_ids['mask_id'])

            # Get text embeddings from ModernBERT
            t_emb = self.bert(input_ids=token_ids_t, attention_mask=seq_mask_t)

            if first == 'motion':
                src_mask = torch.cat([seq_mask_m, seq_mask_t], dim=1)
            elif first == 'text':
                src_mask = torch.cat([seq_mask_t, seq_mask_m], dim=1)
            else:
                raise RuntimeError(f'The order of the two modalities is not assigned.')

            _, pred_t = self.forward(token_ids_m,
                                     src_mask=src_mask,
                                     att_txt=torch.empty(batch_size, 0, dtype=torch.bool, device=lens_t.device),  # empty mask for CLS token
                                     word_emb=t_emb,
                                     first=first,
                                     max_m=max_m,
                                     max_t=max_t)  # (bs, max_t, vocab)

            if rand_pos:
                temperature = 1  # starting_temperature * (steps_until_x0 / timesteps)  # temperature is annealed
            else:
                temperature = 0  # starting_temperature * (steps_until_x0 / timesteps)  # temperature is annealed

            # if temperature == 0, it is equal to argmax (pred_ids = pred_m.argmax(dim=-1))
            pred_ids_t = gumbel_sample(pred_t, temperature=temperature, dim=-1)
            is_mask = token_ids_t == special_ids['mask_id']

            token_ids_t = torch.where(is_mask, pred_ids_t, token_ids_t)

        return token_ids_t

    def inpaint(self, first_tokens, last_tokens, clip_feature=None, word_emb=None, inpaint_len=2, rand_pos=False):
        # support only one sample
        assert first_tokens.shape[0] == 1
        assert last_tokens.shape[0] == 1
        max_steps = 20
        max_length = 49
        batch_size = first_tokens.shape[0]
        mask_id = self.num_vq + 2
        pad_id = self.num_vq + 1
        end_id = self.num_vq
        shape = (batch_size, self.block_size - 1)
        scores = torch.ones(shape, dtype = torch.float32, device = first_tokens.device)
        
        # force add first / last tokens
        first_partition_pos_idx = first_tokens.shape[1]
        second_partition_pos_idx = first_partition_pos_idx + inpaint_len
        end_pos_idx = second_partition_pos_idx + last_tokens.shape[1]

        m_tokens_len = torch.ones(batch_size, device = first_tokens.device)*end_pos_idx

        src_token_mask = generate_src_mask(self.block_size-1, m_tokens_len+1)
        src_token_mask_noend = generate_src_mask(self.block_size-1, m_tokens_len)
        ids = torch.full(shape, mask_id, dtype = torch.long, device = first_tokens.device)
        
        ids[:, :first_partition_pos_idx] = first_tokens
        ids[:, second_partition_pos_idx:end_pos_idx] = last_tokens
        src_token_mask_noend[:, :first_partition_pos_idx] = False
        src_token_mask_noend[:, second_partition_pos_idx:end_pos_idx] = False
        
        # [TODO] confirm that these 2 lines are not neccessary (repeated below and maybe don't need them at all)
        ids[~src_token_mask] = pad_id # [INFO] replace with pad id
        ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # [INFO] replace with end id

        temp = []
        sample_max_steps = torch.round(max_steps/max_length*m_tokens_len) + 1e-8

        if clip_feature is None:
            clip_feature = torch.zeros(1, 512).to(first_tokens.device)
            att_txt = torch.zeros((batch_size,1), dtype=torch.bool, device = first_tokens.device)
        else:
            att_txt = torch.ones((batch_size,1), dtype=torch.bool, device = first_tokens.device)

        for step in range(max_steps):
            timestep = torch.clip(step/(sample_max_steps), max=1)
            rand_mask_prob = cosine_schedule(timestep) # timestep #
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)
            # [INFO] rm no motion frames
            scores[~src_token_mask_noend] = 0
            # [INFO] rm begin and end frames
            scores[:, :first_partition_pos_idx] = 0
            scores[:, second_partition_pos_idx:end_pos_idx] = 0
            scores = scores/scores.sum(-1)[:, None] # normalize only unmasked token
            
            sorted, sorted_score_indices = scores.sort(descending=True) # deterministic
            
            ids[~src_token_mask] = pad_id # [INFO] replace with pad id
            ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # [INFO] replace with end id
            ## [INFO] Replace "mask_id" to "ids" that have highest "num_token_masked" "scores" 
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
            # [INFO] repeat last_id to make it scatter_ the existing last ids.
            last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1)-1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index*~select_masked_indices)
            ids.scatter_(-1, sorted_score_indices, mask_id)

            # [TODO] force replace begin/end tokens b/c the num mask will be more than actual inpainting frames
            ids[:, :first_partition_pos_idx] = first_tokens
            ids[:, second_partition_pos_idx:end_pos_idx] = last_tokens
            
            logits = self.forward(ids, clip_feature, src_token_mask, word_emb=word_emb)[:,1:]
            filtered_logits = logits #top_k(logits, topk_filter_thres)
            if rand_pos:
                temperature = 1 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed
            else:
                temperature = 0 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            # [INFO] if temperature==0: is equal to argmax (filtered_logits.argmax(dim = -1))
            # pred_ids = filtered_logits.argmax(dim = -1)
            pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            is_mask = ids == mask_id
            temp.append(is_mask[:1])
            
            ids = torch.where(
                        is_mask,
                        pred_ids,
                        ids
                    )
            
            probs_without_temperature = logits.softmax(dim = -1)
            scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')
            scores = scores.masked_fill(~is_mask, 0)
        return ids

class Attention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x, src_mask):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if src_mask is not None:
            att[~src_mask] = float('-inf')
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, src_mask=None):
        x = x + self.attn(self.ln1(x), src_mask)
        x = x + self.mlp(self.ln2(x))
        return x
    
class CrossAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, 77)).view(1, 1, block_size, 77))
        self.n_head = n_head

    def forward(self, x,word_emb):
        B, T, C = x.size()
        B, N, D = word_emb.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(word_emb).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(word_emb).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, N) -> (B, nh, T, N)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, N) x (B, nh, N, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block_crossatt(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.attn = CrossAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x,word_emb):
        x = x + self.attn(self.ln1(x), self.ln3(word_emb))
        x = x + self.mlp(self.ln2(x))
        return x

class CrossCondTransBase(nn.Module):

    def __init__(self, 
                vqvae,
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                num_local_layer = 1,
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.vqvae = vqvae
        # self.tok_emb = nn.Embedding(num_vq + 3, embed_dim).requires_grad_(False) 
        self.learn_tok_emb = nn.Embedding(3, self.vqvae.vqvae.code_dim)# [INFO] 3 = [end_id, blank_id, mask_id]
        self.to_emb = nn.Linear(self.vqvae.vqvae.code_dim, embed_dim)

        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers-num_local_layer)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.num_local_layer = num_local_layer
        self.word_emb = nn.Linear(clip_dim, embed_dim)
        if num_local_layer > 0:
            self.cross_att = nn.Sequential(*[Block_crossatt(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_local_layer)])
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, clip_feature, src_mask, word_emb, first=None):
        if len(idx) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            not_learn_idx = idx<self.vqvae.vqvae.num_code
            learn_idx = ~not_learn_idx
            
            token_embeddings = torch.empty((*idx.shape, self.vqvae.vqvae.code_dim), device=idx.device)
            token_embeddings[not_learn_idx] = self.vqvae.vqvae.quantizer.dequantize(idx[not_learn_idx]).requires_grad_(False) 
            token_embeddings[learn_idx] = self.learn_tok_emb(idx[learn_idx]-self.vqvae.vqvae.num_code)
            token_embeddings = self.to_emb(token_embeddings)  # (bs, max_m, embed_dim)

            if first == 'motion':
                word_emb = self.word_emb(word_emb)                                         # (bs, max_t, embed_dim)
                token_embeddings = torch.cat([token_embeddings, word_emb], dim=1)  # (bs, max_m + max_t, embed_dim)
            elif first == 'text':
                word_emb = self.word_emb(word_emb)                                         # (bs, max_t, embed_dim)
                token_embeddings = torch.cat([word_emb, token_embeddings], dim=1)  # (bs, max_t + max_m, embed_dim)
            else:
                # Original MMM
                if self.num_local_layer > 0:
                    word_emb = self.word_emb(word_emb)                         # (bs, max_t, embed_dim)
                    token_embeddings = self.pos_embed(token_embeddings)        # (bs, max_m, embed_dim)
                    # text tokens - (cross attn) - motion tokens --> attn tokens
                    for module in self.cross_att:
                        token_embeddings = module(token_embeddings, word_emb)  # (bs, max_m, embed_dim)
                # concat [text EOS token, attn tokens]
                token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)

        x = self.pos_embed(token_embeddings)
        for block in self.blocks:
            x = block(x, src_mask)

        return x


class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024,
                num_vt=0,
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq, bias=False)
        if num_vt > 0:
            self.head_t = nn.Linear(embed_dim, num_vt, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, src_mask, first=None, max_m=None, max_t=None):
        for block in self.blocks:
            x = block(x, src_mask)
        x = self.ln_f(x)
        if first == 'motion':
            logits_m = self.head(x[:, :max_m])    # (bs, max_m, vocab)
            logits_t = self.head_t(x[:, max_m:])  # (bs, max_t, vocab)
            return logits_m, logits_t
        elif first == 'text':
            logits_t = self.head_t(x[:, :max_t])  # (bs, max_t, vocab)
            logits_m = self.head(x[:, max_t:])    # (bs, max_m, vocab)
            return logits_m, logits_t
        else:
            logits = self.head(x)
            return logits