import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.conv_type import STRConv, STRConvMask

class CosineDecay(object):
    def __init__(self, prune_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, prune_rate=None):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, prune_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, prune_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return prune_rate*self.factor
        else:
            return prune_rate

class GraNet(object):
    def __init__(self, optimizer, prune_rate, prune_rate_decay, death_mode, growth_mode, redistribution_mode, args, train_loader, device, growth_death_ratio=1.0):
        # super(GraNet, self).__init__()
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))
        self.optimizer = optimizer
        self.prune_rate = prune_rate
        self.growth_death_ratio = growth_death_ratio
        self.prune_rate_decay = prune_rate_decay
        self.death_mode = death_mode
        self.growth_mode = growth_mode
        self.redistribution_mode = redistribution_mode
        self.args = args
        self.train_loader = train_loader
        self.sparse_init = args.sparse_init

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if self.args.fix:
            self.prune_every_k_steps = None
        else:
            self.prune_every_k_steps = self.args.update_frequency
        
        self.masks = {}
        self.names = []
        self.modules = []

        self.steps = 0
        self.total_params = 0

    def init(self, mode='ERK', density=0.05, erk_power_scale=1.0, grad_dict=None):
        if self.sparse_init == 'balanced':
            # Logic for Balanced Initialization and Mask Creation
            pass
        elif self.sparse_init == 'ERK':
            # Logic for ERK Initialization and Mask Creation
            print('initialize by ERK')
            for name, weight in self.masks.items():
                self.total_params += weight.numel()
                if 'classifier' in name:
                    self.fc_params = weight.numel()
            is_epsilon_valid = False
            dense_layers = set()
            while not is_epsilon_valid:

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    if name in dense_layers:
                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= n_zeros

                    else:
                        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                        # equation above.
                        rhs += n_ones
                        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale
                        # Note that raw_probabilities[mask] * n_param gives the individual
                        # elements of the divisor.
                        divisor += raw_probabilities[name] * n_param
                # By multipliying individual probabilites with epsilon, we should get the
                # number of parameters per layer correctly.
                epsilon = rhs / divisor
                # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
                # mask to 0., so they become part of dense_layers sets.
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

                total_nonzero += density_dict[name] * mask.numel()
            print(f"Overall sparsity {total_nonzero / self.total_params}")
            
            self.apply_mask()

    def add_module(self, module, sparse_init='ERK', grad_dic=None):
        self.module = module
        self.sparse_init = sparse_init
        self.modules.append(module)
        for name, tensor in module.named_parameters():
            # Only apply mask on 4D layers i.e. Conv Weights
            if len(tensor.size()) == 4 or len(tensor.size()) == 2:  # Change to isInstance(STRConvMask)?
                self.names.append(name)
                self.masks[name] = torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).cuda()

        if self.args.rm_first: # rm_first decides whether to keep the first layer dense or sparse
            for name, tensor in module.named_parameters():
                if 'conv.weight' in name or 'feature.0.weight' in name:
                    self.masks.pop(name)
                    print(f"pop out {name}")

        self.init(mode=self.args.sparse_init, density=self.args.init_density, grad_dict=grad_dic)

    def apply_mask(self):
        for name, tensor in self.module.named_parameters():
            if len(tensor.size()) == 4 or len(tensor.size()) == 2:
                tensor.data = tensor.data * self.masks[name]

    def step(self):
        self.optimizer.step()
        self.apply_mask()
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr(self.prune_rate)
        self.steps += 1

        if self.prune_every_k_steps is not None:
            if self.args.method == 'GraNet':
                if self.steps >= (self.args.init_prune_epoch * len(self.train_loader)) and self.steps % self.prune_every_k_steps == 0:
                    # We do not have to prune since continuous sparsification is being done already
                    self.truncate_weights(self.steps)
                    self.print_nonzero_counts()

    def truncate_weights(self, step):
        # Logic for Truncating Weights
        print('Truncating Weights')
        pass

    def print_nonzero_counts(self):
        # Logic for Printing Nonzero Counts
        print('Printing Nonzero Counts')
        for name, module in self.module.named_modules():
            if isinstance(module, STRConvMask) or isinstance(module, STRConv):
                num_nonzero_params = torch.nonzero(module.weight.data).size(0)
                print(f"{name}: {num_nonzero_params}")
        pass

    
