import argparse
import sys
import yaml

from configs import parser as _parser

args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training for STR, DNW and GMP")

    # General Config
    parser.add_argument(
        "--data", help="path to dataset base directory", default=""
    )
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--set", help="name of dataset", type=str, default="ImageNet")
    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="ResNet18", help="model architecture"
    )
    
    # parser.add_argument(
    #     "--config", help="Config file to use (see configs dir)", default=None
    # )
    parser.add_argument(
        "--config", help="Config file to use (see configs dir)", default="configs/largescale/resnet18-str.yaml"
    )
    parser.add_argument(
        "--log-dir", help="Where to save the runs. If None use ./runs", default=None
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=20,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 20)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=None,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--warmup_length", default=0, type=int, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--init_prune_epoch", default=0, type=int, help="Init epoch for pruning in GMP"
    )
    parser.add_argument(
        "--final_prune_epoch", default=-100, type=int, help="Final epoch for pruning in GMP"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--num-classes",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use for multigpu training",
    )

    # Learning Rate Policy Specific
    parser.add_argument(
        "--lr-policy", default="constant_lr", help="Policy for the learning rate."
    )
    parser.add_argument(
        "--multistep-lr-adjust", default=30, type=int, help="Interval to drop lr"
    )
    parser.add_argument(
        "--multistep-lr-gamma", default=0.1, type=int, help="Multistep multiplier"
    )
    parser.add_argument(
        "--name", default=None, type=str, help="Experiment name to append to filepath"
    )
    parser.add_argument(
        "--save_every", default=-1, type=int, help="Save every ___ epochs"
    )
    parser.add_argument(
        "--prune-rate",
        default=0.0,
        help="Amount of pruning to do during sparse training",
        type=float,
    )
    parser.add_argument(
        "--width-mult",
        default=1.0,
        help="How much to vary the width of the network.",
        type=float,
    )
    parser.add_argument(
        "--nesterov",
        default=False,
        action="store_true",
        help="Whether or not to use nesterov for SGD",
    )
    parser.add_argument(
        "--random-mask",
        action="store_true",
        help="Whether or not to use a random mask when fine tuning for lottery experiments",
    )
    parser.add_argument(
        "--one-batch",
        action="store_true",
        help="One batch train set for debugging purposes (test overfitting)",
    )
    parser.add_argument(
        "--conv-type", type=str, default=None, help="What kind of sparsity to use"
    )
    parser.add_argument(
        "--freeze-weights",
        action="store_true",
        help="Whether or not to train only mask (this freezes weights)",
    )
    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
    parser.add_argument("--bn-type", default=None, help="BatchNorm type")
    parser.add_argument(
        "--init", default="kaiming_normal", help="Weight initialization modifications"
    )
    parser.add_argument(
        "--no-bn-decay", action="store_true", default=False, help="No batchnorm decay"
    )
    parser.add_argument(
        "--dense-conv-model", action="store_true", default=False, help="Store a model variant of the given pretrained model that is compatible to CNNs with DenseConv (nn.Conv2d)"
    )
    parser.add_argument(
        "--st-decay", type=float, default=None, help="decay for sparse thresh. If none then use normal weight decay."
    )
    parser.add_argument(
        "--scale-fan", action="store_true", default=False, help="scale fan"
    )
    parser.add_argument(
        "--first-layer-dense", action="store_true", help="First layer dense or sparse"
    )
    parser.add_argument(
        "--last-layer-dense", action="store_true", help="Last layer dense or sparse"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        help="Label smoothing to use, default 0.0",
        default=None,
    )
    parser.add_argument(
        "--first-layer-type", type=str, default=None, help="Conv type of first layer"
    )

    parser.add_argument(
        "--sInit-type",
        type=str,
        help="type of sInit",
        default="constant",
    )
    
    parser.add_argument(
        "--sInit-value",
        type=float,
        help="initial value for sInit",
        default=100,
    )

    parser.add_argument(
        "--sparse-function", type=str, default='sigmoid', help="choice of g(s)"
    )

    parser.add_argument(
        "--use-budget", action="store_true", help="use the budget from the pretrained model."
    )
    parser.add_argument(
        "--ignore-pretrained-weights", action="store_true", help="ignore the weights of a pretrained model."
    )

    # Dynamic Sparse Training related arguments
    parser.add_argument(
        '--growth', type=str, default='gradient', help='Growth mode. Choose from: momentum, random, and momentum_neuron.'
        )
    parser.add_argument(
        '--prune', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold, CS_death.'
        )
    parser.add_argument(
        '--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.'
        )
    parser.add_argument(
        '-- -rate', type=float, default=0.50, help='The pruning rate / death rate for Zero-Cost Neuroregeneration.'
        )
    parser.add_argument(
        '--pruning-rate', type=float, default=0.50, help='The pruning rate / death rate.'
        )
    parser.add_argument(
        '--sparse', action='store_true', help='Enable sparse mode. Default: True.'
        )
    parser.add_argument(
        '--fix', action='store_true', help='Fix topology during training. Default: True.'
        )
    parser.add_argument(
        '--update-frequency', type=int, default=100, metavar='N', help='how many iterations to train between mask update'
        )
    parser.add_argument(
        '--sparse-init', type=str, default='ERK, uniform distributions for sparse training, global pruning and uniform pruning for pruning', help='sparse initialization'
        )
    parser.add_argument(
        '--dst-prune-const', action='store_true', help='If true, prune rate is constant at args.final_density. Else, prune rate is annealed between args.init_density and args.final_density.'
        )
    
    # hyperparameters for gradually pruning
    parser.add_argument(
        '--method', type=str, default='GraNet', help='method name: DST, GraNet, GraNet_uniform, GMP, GMO_uniform'
        )
    parser.add_argument(
        '--init-density', type=float, default=0.0, help='The pruning rate / death rate for sparse pruning. Range: (0, 1).'
        )
    parser.add_argument(
        '--final-density', type=float, default=0.05, help='The density of the overall sparse network. Must be less than args.init_density. Range: (0, 1).'
        )
    parser.add_argument(
        '--const-prune-rate', type=float, default=0.05, help='If args.dst_prune_const is True, pruning is done at a constant rate. Range: (0, 1).'
        )
    parser.add_argument(
        '--init-prune-epoch', type=int, default=0, help='The initial epoch for which pruning is done. Can be called a warmup period'
        )
    parser.add_argument(
        '--final-prune-epoch', type=int, default=110, help='The final epoch for which pruning is done.'
        )
    parser.add_argument(
        '--rm-first', action='store_true', help='Keep the first layer dense.'
        )

    args = parser.parse_args()

    get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
