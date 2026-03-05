import os
import torch
from os.path import join as pjoin
from transformers import AutoTokenizer

from options.option_residual import TrainResTransOptions
from options.get_eval_option import get_opt

from dataset import dataset_TM_train, dataset_TM_eval

from models.vq.model import RVQVAE
from models.residual import ResidualTransformer, ResidualTransformerTrainer
from models.evaluator_wrapper import EvaluatorModelWrapper

from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain
from utils.word_vectorizer import WordVectorizer
from exit.utils import fixseed


def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    net = RVQVAE(vq_opt,
                 dim_pose,
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
    print(f'Loading vq model checkpoint from {resume_pth} ...')
    ckpt = torch.load(resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['vq_model' if 'vq_model' in ckpt else 'net'])
    net.to(opt.device)
    return net, vq_opt

if __name__ == '__main__':
    parser = TrainResTransOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device('cpu' if opt.gpu_id == -1 else 'cuda:' + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/res/', opt.dataset_name, opt.name)
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = t2m_kinematic_chain
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_len = 55
        kinematic_chain = kit_kinematic_chain
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'

    else:
        raise KeyError('Dataset Does Not Exist')

    opt.text_dir = pjoin(opt.data_root, 'texts')

    vq_model, vq_opt = load_vq_model()
    opt.num_tokens = vq_opt.nb_code
    opt.num_quantizers = vq_opt.num_quantizers

    ##### ---- GloVe and BERT Tokenizer ---- #####
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    bert_name = 'google-bert/bert-large-uncased'
    tokenizer = AutoTokenizer.from_pretrained(bert_name)

    ##### ---- Residual Transformer ---- #####
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                          cond_mode='text',
                                          latent_dim=opt.latent_dim,
                                          ff_size=opt.ff_size,
                                          num_layers=opt.n_layers,
                                          num_heads=opt.n_heads,
                                          dropout=opt.dropout,
                                          shared_codebook=vq_opt.shared_codebook,
                                          cond_drop_prob=opt.cond_drop_prob,
                                          share_weight=opt.share_weight,
                                          opt=opt)
    all_params = 0
    pc_transformer = sum(param.numel() for param in res_transformer.parameters_wo_clip())
    print(res_transformer)
    all_params += pc_transformer
    print(f'Total parameters of all models: {pc_transformer / 1_000_000:.2f}M')

    ##### ---- Dataloader ---- #####
    train_loader = dataset_TM_train.DATALoaderNew(opt.dataset_name, '', 'motion_vecs', vq_opt.nb_code,
                                                  batch_size=opt.batch_size, unit_length=2 ** vq_opt.down_t)
    val_loader = dataset_TM_eval.DATALoaderNew(opt.dataset_name, '', w_vectorizer, 'motion_vecs', vq_opt.nb_code,
                                               batch_size=32, is_test=False, tokenizer_t=tokenizer, max_t=opt.max_t)

    ##### ---- Evaluation Wrapper ---- #####
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Trainer ---- #####
    trainer = ResidualTransformerTrainer(opt, res_transformer, vq_model, tokenizer)
    trainer.train(train_loader, val_loader, eval_wrapper=eval_wrapper)