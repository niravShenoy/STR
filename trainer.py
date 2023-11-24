import time
import torch
import tqdm
import math

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
from utils.conv_type import STRConv, STRConvMask


__all__ = ["train", "validate"]

def magnitude_death(mask, weight, prune_rate, num_nonzeros, num_zeros):
    num_remove = math.ceil(prune_rate * num_nonzeros)
    num_retain = math.ceil(num_nonzeros - num_remove)  # [(1 - prune_rate) * num_nonzeros]

    if num_remove == 0.0:
        return weight.data != 0.0
    k = math.ceil(num_zeros + num_remove)

    x1, _ = torch.sort(torch.abs(weight.data[mask == 1.0]), descending=True) # Alternate logic to extract the top [(1 - prune_rate) * num_nonzeros] elements
    threshold1 = x1[num_retain].item()

    x, _ = torch.sort(torch.abs(weight.data.view(-1)))      # Method used in GraNet implementation
    threshold = x[k-1].item()

    assert threshold1 == threshold, f"Threshold mismatch {threshold1} != {threshold}"

    return (torch.abs(weight.data) > threshold1).float()

def gradient_growth(mask, weight, num_pruned):
    grad = weight.grad
    grad = grad * (mask == 0.0).float()
    x, _ = torch.sort(torch.abs(grad.view(-1)), descending=True)
    threshold = x[num_pruned].item()
    # assert threshold > 0.0, f"Threshold is: {threshold}"
    grad_mask = (torch.abs(grad) > threshold).float()
    # assert grad_mask.sum().item() == num_pruned, f"Number of elements in grad_mask {grad_mask.sum().item()} != {num_pruned}"
    return grad_mask

def pruning(model, args, step, train_loader_len):
    """
    Prunes the model based on the defined arguments and current training step. Retains the weights but updates the mask.

    Args:
        model (torch.nn.Module): The model to be pruned.
        args (argparse.Namespace): The arguments containing pruning parameters.
        step (int): The current training step.

    Returns:
        None
    """

    curr_prune_iter = step // args.update_frequency
    final_iter = int((args.final_prune_epoch * train_loader_len) / args.update_frequency)
    init_iter = int((args.init_prune_epoch * train_loader_len) / args.update_frequency)
    total_iter = final_iter - init_iter

    # Conditions for Pruning. Omit if you want to prune based on epochs instead of steps
    assert args.init_density > args.final_density, 'Initial density must be greater than final density'

    if args.dst_prune_const:    # Whether to prune at a constant rate or anneal the pruning rate based on iteration
        # Use a constant Dynamic Prune Rate
        curr_prune_rate = args.const_prune_rate
    else:
        # Using a version of eqn. 1 from section 4.1 the GraNet paper
        if curr_prune_iter >= init_iter and curr_prune_iter <= final_iter - 1:
            prune_decay = (1 - ((curr_prune_iter - init_iter) / total_iter)) ** 3
            curr_prune_rate = (1 - args.init_density) + (args.init_density - args.final_density) * (
                    1 - prune_decay)
        else:
            return
    
    weight_abs = []
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvMask)):
            weight_abs.append(torch.abs(m.weight))

    all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
    num_params_to_keep = int(len(all_scores) * (1 - curr_prune_rate))

    x, _ = torch.topk(all_scores, num_params_to_keep)
    threshold = x[-1]

    total_size = 0
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvMask)):
            total_size += torch.nonzero(m.weight.data).size(0)
            zero = torch.tensor([0.]).to(m.weight.device)
            one = torch.tensor([1.]).to(m.weight.device)
            m.mask = torch.where((torch.abs(m.weight) > threshold), one, zero)

    print('Total Model parameters:', total_size)

    sparse_size = 0
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvMask)):
            sparse_size += torch.nonzero(m.mask).size(0)

    print('% Pruned: {0}'.format((total_size - sparse_size) / total_size))

def truncate_weights(model, prune_rate):
    """
    Creates an updated mask based on the weights of the model using the Prune and Grow algorithm.

    Args:
        model (nn.Module): The model to truncate the weights of.
        args: Additional arguments.
        step: The current step.
        prune_rate (float): The rate at which to prune the weights.

    Returns:
        None
    """

    num_nonzeros = []
    num_zeros = []

    # Model statistics before Prune and Grow
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvMask)):
            num_nonzeros.append(m.mask.sum().item())
            num_zeros.append(m.mask.numel() - num_nonzeros[-1])

    print(f"(Prune and Regrow) Prune Rate: {prune_rate}")
    # Prune
    layer = 0
    num_pruned = []
    updated_mask = []
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvMask)):
            mask = m.mask
            new_mask = magnitude_death(mask, m.weight, prune_rate, num_nonzeros[layer], num_zeros[layer])
            num_pruned.append(int(num_nonzeros[layer] - new_mask.sum().item()))
            updated_mask.append(new_mask)
            m.mask = new_mask
            layer += 1

    # Grow
    layer = 0
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvMask)):
            mask = m.mask
            assert torch.equal(mask, updated_mask[layer]), f"Mask mismatch {mask.sum().item()} != {updated_mask[layer].sum().item()}"

            new_mask = gradient_growth(mask, m.weight, num_pruned[layer])
            m.mask = new_mask + mask

            # Sanity Checks
            assert torch.all(m.mask <= 1.0), "Mask value greater than 1.0"
            # assert m.mask.sum().item() - updated_mask[layer].sum().item() == num_pruned[layer], f"Layer {layer}: Name:{n} -> Pruning and Regeneration mismatch. {m.mask.sum().item()} != {num_nonzeros[layer]}"

            print(f"{n}: Density: {m.mask.sum().item() / m.mask.numel()}")
            layer += 1


def step(model, optimizer, args, step, train_loader_len, decay_scheduler=None):
    optimizer.step()
    # Get prune rate based on CosineDecay or LinearDecay
    # Cosine Decay step
    if decay_scheduler is not None:
        decay_scheduler.step()
        prune_rate = decay_scheduler.get_dr()
    
    # Condition needs to be modified (in value and in logic) if we prune based on epochs instead of steps
    if args.update_frequency is not None:
        if args.method == 'GraNet':
            # Set up a warmup period since we want GraNet to come into the picture in the high sparsity regime
            if step >= (args.init_prune_epoch * train_loader_len) and step % args.update_frequency == 0:
                # Do we perform pruning if STR is anyway going to prune?
                pruning(model, args, step, train_loader_len)
                truncate_weights(model, prune_rate)
                


def train(train_loader, model, criterion, optimizer, epoch, args, writer, decay_scheduler=None):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    batch_len = len(train_loader)
    end = time.time()
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=batch_len
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        loss = criterion(output, target.view(-1))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.sparse:
            # Can avoid steps and use epochs instead (do every 5-10 epochs)
            # This means the logic in step() must be changed to use epochs instead of steps
            num_steps = (epoch * batch_len) + i
            step(model, optimizer, args, num_steps, batch_len, decay_scheduler)
        else:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (batch_len * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True).long()

            # compute output
            output = model(images)

            loss = criterion(output, target.view(-1))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg

