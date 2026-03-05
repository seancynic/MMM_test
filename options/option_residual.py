import os
import argparse
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument("--max-t", type=int, default=77, help="maximum length of text")

        self.parser.add_argument('--name', type=str, default="t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns", help='Name of this trial')
        self.parser.add_argument('--vq-name', type=str, default="rvq_nq1_dc512_nc512", help='Name of the rvq model.')

        self.parser.add_argument("--gpu-id", type=int, default=-1, help='GPU id')
        self.parser.add_argument('--dataset-name', type=str, default='t2m', help='Dataset Name, {t2m} for humanml3d, {kit} for kit-ml')
        self.parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints', help='models are saved here.')

        self.parser.add_argument('--latent-dim', type=int, default=384, help='Dimension of transformer latent.')
        self.parser.add_argument('--n-heads', type=int, default=6, help='Number of heads.')
        self.parser.add_argument('--n-layers', type=int, default=8, help='Number of attention layers.')
        self.parser.add_argument('--ff-size', type=int, default=1024, help='FF_Size')
        self.parser.add_argument('--dropout', type=float, default=0.2, help='Dropout ratio in transformer')

        self.parser.add_argument("--max-motion-length", type=int, default=196, help="Max length of motion")
        self.parser.add_argument("--unit-length", type=int, default=4, help="Downscale ratio of VQ")

        self.parser.add_argument('--force-mask', action="store_true", help='True: mask out conditions')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        self.opt.is_train = self.is_train

        if self.opt.gpu_id != -1:
            # self.opt.gpu_id = int(self.opt.gpu_id)
            torch.cuda.set_device(self.opt.gpu_id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        if self.is_train:
            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.name)
            if not os.path.exists(expr_dir):
                os.makedirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        return self.opt


class TrainResTransOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
        self.parser.add_argument('--max-epoch', type=int, default=500, help='Maximum number of epoch for training')

        '''LR scheduler'''
        self.parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate schedule factor')
        self.parser.add_argument('--milestones', default=[50_000], nargs="+", type=int,
                            help="learning rate schedule (iterations)")
        self.parser.add_argument('--warm-up-iter', default=2000, type=int, help='number of total iterations for warmup')

        '''Condition'''
        self.parser.add_argument('--cond-drop-prob', type=float, default=0.1, help='Drop ratio of condition, for classifier-free guidance')
        self.parser.add_argument("--seed", default=3407, type=int, help="Seed")

        self.parser.add_argument('--is-continue', action="store_true", help='Is this trial continuing previous state?')
        self.parser.add_argument('--gumbel-sample', action="store_true", help='Strategy for token sampling, True: Gumbel sampling, False: Categorical sampling')
        self.parser.add_argument('--share-weight', action="store_true", help='Whether to share weight for projection/embedding, for residual transformer.')

        self.parser.add_argument('--log-every', type=int, default=50, help='Frequency of printing training progress, (iteration)')
        self.parser.add_argument('--eval-every-e', type=int, default=10, help='Frequency of animating eval results, (epoch)')
        self.parser.add_argument('--save-latest', type=int, default=500, help='Frequency of saving checkpoint, (iteration)')

        self.is_train = True