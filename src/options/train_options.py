from .base_options import BaseOptions

# Here is the options especially for training

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--log_dir', type=str, default='./logs', help='the path to record log')
        parser.add_argument('--epoch_num_steps', type=int, default=5000, help='number of steps in each epoch')
        parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=50, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--print_freq', type=int, default=500, help='frequency of showing training results on console')
        parser.add_argument('--display_freq', type=int, default=500, help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--which_epoch', type=str, default='', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy: linear|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.isTrain = True
        return parser