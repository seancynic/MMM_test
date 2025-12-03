import torch
import clip
import models.vqvae as vqvae
from models.vqvae_sep import VQVAE_SEP
import models.t2m_trans as trans
import models.t2m_trans_uplow as trans_uplow
import numpy as np
from exit.utils import visualize_2motions
import options.option_transformer as option_trans
from transformers import AutoTokenizer, ModernBertModel


max_m = 50
max_t = 77
first_modality = 'motion'  # "motion" or "text"

##### ---- CLIP ---- #####
device = torch.device('cuda')
model_name = 'answerdotai/modernbert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
modernbert = ModernBertModel.from_pretrained(model_name, attn_implementation='eager').to(device).half()  # float16
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


def get_vqvae(args, is_upper_edit):
    if not is_upper_edit:
        return vqvae.HumanVQVAE(args,  ## use args to define different parameters in different quantizers
                                args.nb_code,
                                args.code_dim,
                                args.output_emb_width,
                                args.down_t,
                                args.stride_t,
                                args.width,
                                args.depth,
                                args.dilation_growth_rate)
    else:
        return VQVAE_SEP(args,  ## use args to define different parameters in different quantizers
                         args.nb_code,
                         args.code_dim,
                         args.output_emb_width,
                         args.down_t,
                         args.stride_t,
                         args.width,
                         args.depth,
                         args.dilation_growth_rate,
                         moment={'mean': torch.from_numpy(args.mean).cuda().float(),
                                 'std': torch.from_numpy(args.std).cuda().float()},
                         sep_decoder=True)


def get_maskdecoder(args, vqvae, is_upper_edit):
    tranformer = trans if not is_upper_edit else trans_uplow
    return tranformer.Text2Motion_Transformer(vqvae,
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


class MMM(torch.nn.Module):
    def __init__(self, args=None, is_upper_edit=False):
        super().__init__()
        self.is_upper_edit = is_upper_edit

        args.dataname = args.dataset_name = 't2m'

        self.vqvae = get_vqvae(args, is_upper_edit)
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        self.vqvae.load_state_dict(ckpt['net'], strict=True)
        if is_upper_edit:
            class VQVAE_WRAPPER(torch.nn.Module):
                def __init__(self, vqvae):
                    super().__init__()
                    self.vqvae = vqvae

                def forward(self, *args, **kwargs):
                    return self.vqvae(*args, **kwargs)

            self.vqvae = VQVAE_WRAPPER(self.vqvae)
        self.vqvae.eval()
        self.vqvae.cuda()

        self.maskdecoder = get_maskdecoder(args, self.vqvae, is_upper_edit)
        ckpt = torch.load(args.resume_trans, map_location='cpu')
        self.maskdecoder.load_state_dict(ckpt['trans'], strict=True)
        self.maskdecoder.eval()
        self.maskdecoder.cuda()

    def forward(self, text, lengths=None, rand_pos=True):
        ## TODO: motion to text
        ##### Only for text to motion #####
        target_len = torch.ceil(lengths / 4).int()  # target token length

        # encode text
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_t, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        input_attn_mask = inputs['attention_mask'].to(device)
        word_emb = bert(input_ids=input_ids, attention_mask=input_attn_mask)

        # generate target tokens
        index_motion = self.maskdecoder(type='sample_new', valid_length=lengths, max_m=max_m, max_t=max_t,
                                        rand_pos=rand_pos, first=first_modality,
                                        word_emb=word_emb, seq_mask_t=input_attn_mask)

        # decode target tokens
        pred_pose_all = torch.zeros((len(text), 196, 263)).cuda()
        for k in range(len(text)):
            pred_pose = self.vqvae(index_motion[k:k + 1, :target_len[k]], type='decode')
            pred_pose_all[k:k + 1, :int(lengths[k].item())] = pred_pose

        return pred_pose_all


if __name__ == '__main__':
    args = option_trans.get_args_parser()

    mmm = MMM(args).cuda()
    pred_pose = mmm([args.text], torch.tensor([args.length]).cuda(), rand_pos=False)

    std = np.load('./exit/t2m-std.npy')
    mean = np.load('./exit/t2m-mean.npy')
    file_name = '_'.join(args.text.split(' ')) + '_' + str(args.length)
    visualize_2motions(pred_pose[0].detach().cpu().numpy(), std, mean, 't2m', args.length, save_path='./output/' + file_name + '.html')