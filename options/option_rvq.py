import argparse
import os
import torch


def get_opt_parser(is_train=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--dataset-name', type=str, default='t2m', help='Dataset Name, {t2m} for humanml3d, {kit} for kit-ml')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')

    ## optimization
    parser.add_argument('--max-epoch', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--warm-up-iter', default=2000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--milestones', default=[150000, 250000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss-vel', type=float, default=0.5, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons-loss', type=str, default='l1_smooth', help='reconstruction loss')

    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="num of resblocks for each res")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices=['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')

    parser.add_argument('--num-quantizers', type=int, default=6, help='num_quantizers')
    parser.add_argument('--shared-codebook', action="store_true")
    parser.add_argument('--quantize-dropout-prob', type=float, default=0.2, help='quantize_dropout_prob')

    parser.add_argument('--ext', type=str, default='default', help='reconstruction loss')

    ## other
    parser.add_argument('--name', type=str, default="rvqvae", help='Name of this trial')
    parser.add_argument('--is-continue', action="store_true", help='Name of this trial')
    parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--log-every', default=10, type=int, help='iter log frequency')
    parser.add_argument('--save-latest', default=500, type=int, help='iter save latest model frequency')
    parser.add_argument('--save-every-e', default=2, type=int, help='save model every n epoch')
    parser.add_argument('--eval-every-e', default=1, type=int, help='save eval results every n epoch')
    parser.add_argument('--feat-bias', type=float, default=5, help='Layers of GRU')

    parser.add_argument('--which-epoch', type=str, default="all", help='Name of this trial')

    ## For Res Predictor only
    parser.add_argument('--vq-name', type=str, default="rvq_nq6_dc512_nc512_noshare_qdp0.2", help='Name of this trial')
    parser.add_argument("--seed", default=123, type=int)

    opt, _ = parser.parse_known_args()
    # torch.cuda.set_device(opt.gpu_id)

    args = vars(opt)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    opt.is_train = is_train
    if is_train:
    # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

    return opt