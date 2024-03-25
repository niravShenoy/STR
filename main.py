import os
import pathlib
import random
import shutil
import time
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import save_checkpoint, get_lr, LabelSmoothing
from utils.schedulers import get_policy, warmup_lr, assign_learning_rate
from utils.conv_type import STRConv, STRConvMask
from utils.conv_type import sparseFunction
import utils.conv_type as conv_type

from args import args
from trainer import train, validate
from core import GraNet, LinearDecay, CosineDecay

import data
from models import resnet


def main():
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    # args.gpu = None
    args.gpu = torch.cuda.current_device()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model and optimizer
    model = get_model(args)

    # If args.sparse_init not specified, returns a fully dense model
    model = sparseInit(model)

    model = set_gpu(args, model)
    print(f'Model Architecture: {model}')

    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)

    # Loading pretrained model
    if args.pretrained:
        pretrained(args, model)

        # Saving a DenseConv (nn.Conv2d) compatible model 
        if args.dense_conv_model:    
            print(f"==> DenseConv compatible model, saving at {ckpt_base_dir / 'model_best.pth'}")
            save_checkpoint(
                {
                    "epoch": 0,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                },
                True,
                filename=ckpt_base_dir / f"epoch_pretrained.state",
                save=True,
            )
            return

    optimizer = get_optimizer(args, model)
    data = get_dataset(args)

    # if args.label_smoothing is None:
    #     criterion = nn.CrossEntropyLoss().cuda()
    # else:
    #     criterion = LabelSmoothing(smoothing=args.label_smoothing)

    
    criterion = nn.CrossEntropyLoss().cuda()
    decay = None

    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    if args.resume:
        best_acc1 = resume(args, model, optimizer)

    # Evaulation of a model
    if args.evaluate:
        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )
        return

    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None

    warmup_scheduler = warmup_lr(optimizer, args)
    val_acc_total = []

    base_dir = ''

    # if not args.warmup:
    #     args.warmup_length = 0
    # if args.warmup:
    #     print('Warm Up training for the model')
    #     warmup_decay = CosineDecay(args.prune_rate, len(data.train_loader) * args.warmup_length)
    #     for epoch in range(args.warmup_length):
    #         warmup_scheduler(epoch)
    #         lr = get_lr(optimizer)
    #         print('The current learning rate is: ', lr)

    #         start_train = time.time()
    #         train_acc1, train_acc5 = train(
    #             data.train_loader, model, criterion, optimizer, epoch, args, writer=writer, decay_scheduler=warmup_decay
    #         )
    #         train_time.update((time.time() - start_train) / 60)
    #         # evaluate on validation set
    #         start_validation = time.time()
    #         acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
    #         validation_time.update((time.time() - start_validation) / 60)
            
    #         val_acc_total.append(acc1)

    # # save the model and the optimizer
    # torch.save(model.state_dict(), "{}runs/model_{}_init.pt".format(base_dir, args.name))
    # torch.save(optimizer.state_dict(),"{}runs/optimizer_{}.pt".format(base_dir, args.name))
    # torch.save(val_acc_total, 'runs/val_acc_'+ args.name + '.pt')

    # if args.conv_type == "STRConv" or args.conv_type == "STRConvMask":
    #     prune_scheduler = get_policy(args.prune_scheduler)(optimizer, args)

    #     for epoch in range(args.final_prune_epoch):
    #         prune_scheduler(epoch)
    #         lr = get_lr(optimizer)
    #         print('The curent learning rate is: ', lr)

    #         start_train = time.time()
    #         train_acc1, train_acc5 = train(
    #             data.train_loader, model, criterion, optimizer, epoch, args, writer=writer
    #         )
    #         train_time.update((time.time() - start_train) / 60)
    #         # evaluate on validation set
    #         start_validation = time.time()
    #         acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
    #         validation_time.update((time.time() - start_validation) / 60)
    #         val_acc_total.append(acc1)
        
    #     is_best = acc1 > best_acc1
    #     best_acc1 = max(acc1, best_acc1)
    #     best_acc5 = max(acc5, best_acc5)
    #     best_train_acc1 = max(train_acc1, best_train_acc1)
    #     best_train_acc5 = max(train_acc5, best_train_acc5)

    # save the mask of the sparse structure
    mask_list = []
    total_num = 0
    total_den = 0
    for name, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvMask)):
            mask_list.append(m.mask)
            total_num += m.mask.sum()
            total_den += m.mask.numel()
    print('Density before full training is: ', total_num / (total_den + 1e-8))
    torch.save(mask_list, 'runs/mask_{}.pt'.format(args.name))
    

    # Start training
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    # Creating an outer loop for iterative STR training
    for iter in range(args.iterations):
        assign_learning_rate(optimizer, args.lr)
        lr_policy = get_policy(args.lr_policy)(optimizer, args)

        decay = CosineDecay(args.prune_rate, len(data.train_loader) * args.epochs)

        total_num = 0
        total_den = 0
        for _, m in model.named_modules():
            if isinstance(m, (STRConv, STRConvMask)):
                m.sparseThreshold = nn.Parameter(conv_type.initialize_sInit())
                total_num += m.mask.sum()
                total_den += m.mask.numel()
        print(f'Density before STR iteration {iter+1} is: {total_num / (total_den + 1e-8)}')

        for epoch in range(args.epochs):
            lr_policy(epoch, iteration=None)
            cur_lr = get_lr(optimizer)
            print('The curent learning rate is: ', cur_lr)

            # train for one epoch
            start_train = time.time()
            train_acc1, train_acc5 = train(
                data.train_loader, model, criterion, optimizer, epoch, args, writer=writer, decay_scheduler=decay
            )
            train_time.update((time.time() - start_train) / 60)

            # evaluate on validation set
            start_validation = time.time()
            acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
            validation_time.update((time.time() - start_validation) / 60)
            val_acc_total.append(acc1)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            best_acc5 = max(acc5, best_acc5)
            best_train_acc1 = max(train_acc1, best_train_acc1)
            best_train_acc5 = max(train_acc5, best_train_acc5)

            save = ((epoch % args.save_every) == 0) and args.save_every > 0
            if is_best or save or epoch == args.epochs - 1:
                if is_best:
                    print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "best_acc1": best_acc1,
                        "best_acc5": best_acc5,
                        "best_train_acc1": best_train_acc1,
                        "best_train_acc5": best_train_acc5,
                        "optimizer": optimizer.state_dict(),
                        "curr_acc1": acc1,
                        "curr_acc5": acc5,
                    },
                    is_best,
                    filename=ckpt_base_dir / f"epoch_{epoch}.state",
                    save=save,
                )

            epoch_time.update((time.time() - end_epoch) / 60)
            progress_overall.display(epoch)
            progress_overall.write_to_tensorboard(
                writer, prefix="diagnostics", global_step=epoch
            )

            writer.add_scalar("test/lr", cur_lr, epoch)
            end_epoch = time.time()
            torch.save(val_acc_total, 'runs/val_acc_'+ args.name + '.pt') 

            # Storing sparsity and threshold statistics for STRConv models
            if args.conv_type == "STRConv" or args.conv_type == "STRConvMask":
                count = 0.0
                num_nonzeros = 0.0
                for n, m in model.named_modules():
                    if isinstance(m, (STRConv, STRConvMask)):
                        sparsity, total_params, thresh, str_sparsity = m.getSparsity()
                        writer.add_scalar("overall_sparsity/{}".format(n), sparsity, epoch)
                        writer.add_scalar("str_sparsity/{}".format(n), str_sparsity, epoch)
                        writer.add_scalar("thresh/{}".format(n), thresh, epoch)
                        num_nonzeros += int(((100 - sparsity) / 100) * total_params)
                        count += total_params
                total_sparsity = 100 - (100 * num_nonzeros / count)
                writer.add_scalar("overall_sparsity/total", total_sparsity, epoch)
            writer.add_scalar("test/lr", cur_lr, epoch)
            end_epoch = time.time()

    torch.save(model.state_dict(),"runs/model_{}_trained.pt".format(args.name))
    args.prune_rate = 1 - args.init_density

    write_result_to_csv(
        best_acc1=best_acc1,
        best_acc5=best_acc5,
        best_train_acc1=best_train_acc1,
        best_train_acc5=best_train_acc5,
        prune_rate=args.prune_rate,
        curr_acc1=acc1,
        curr_acc5=acc5,
        base_config=args.config,
        name=args.name,
    )
    if args.conv_type == "STRConv" or args.conv_type == "STRConvMask":
        json_data = {}
        json_thres = {}
        num_nonzeros = 0.0
        count = 0.0
        layer = 0
        for n, m in model.named_modules():
            if isinstance(m, (STRConv, STRConvMask)):
                sparsity = m.getSparsity()
                writer.add_scalar("overall_sparsity/layerwise_sparsity_dist", sparsity[0], layer)
                json_data[n] = sparsity[0]
                num_nonzeros += int(((100 - sparsity[0]) / 100) * sparsity[1])
                count += sparsity[1]
                json_thres[n] = sparsity[2]
                writer.add_scalar("thresh/layerwise_threshold_dist", sparsity[2], layer)
                layer += 1
        json_data["total"] = 100 - (100 * num_nonzeros / count)
        if not os.path.exists("runs/layerwise_sparsity"):
            os.mkdir("runs/layerwise_sparsity")
        if not os.path.exists("runs/layerwise_threshold"):
            os.mkdir("runs/layerwise_threshold")
        with open("runs/layerwise_sparsity/{}.json".format(args.name), "w") as f:
            json.dump(json_data, f)
        with open("runs/layerwise_threshold/{}.json".format(args.name), "w") as f:
            json.dump(json_thres, f)


def set_gpu(args, model):
    torch.cuda.empty_cache()
    if args.gpu is not None:
        # torch.cuda.set_device(args.gpu)
        device = torch.device(args.gpu)
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        # torch.cuda.set_device(args.multigpu[0])
        torch.cuda.device(args.multigpu[0])
        # args.gpu = args.multigpu[0]
        args.gpu = torch.cuda.device(args.multigpu[0])
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.gpu
        )

    cudnn.benchmark = True

    return model


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume)
        if args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        print(f"=> No checkpoint found at '{args.resume}'")


def pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )["state_dict"]

        model_state_dict = model.state_dict()

        if not args.ignore_pretrained_weights:

            pretrained_final = {
                k: v
                for k, v in pretrained.items()
                if (k in model_state_dict and v.size() == model_state_dict[k].size())
            }

            if args.conv_type != "STRConv" and args.conv_type != "STRConvMask":
                for k, v in pretrained.items():
                    if 'sparseThreshold' in k:
                        wkey = k.split('sparse')[0] + 'weight'
                        weight = pretrained[wkey]
                        pretrained_final[wkey] = sparseFunction(weight, v)

            model_state_dict.update(pretrained_final)
            model.load_state_dict(model_state_dict)

        # Using the budgets of STR models for other models like DNW and GMP
        if args.use_budget:
            budget = {}
            for k, v in pretrained.items():
                if 'sparseThreshold' in k:
                    wkey = k.split('sparse')[0] + 'weight'
                    weight = pretrained[wkey]
                    sparse_weight = sparseFunction(weight, v)
                    budget[wkey] = (sparse_weight.abs() > 0).float().mean().item()

            for n, m in model.named_modules():
                if hasattr(m, 'set_prune_rate'):
                    pr = 1 - budget[n + '.weight']
                    m.set_prune_rate(pr)
                    print('set prune rate', n, pr)


    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(args.arch))
    # model = models.__dict__[args.arch]()
    model = None
    if args.set == "CIFAR10":
        model = resnet.ResNetWidth18([3, 32, 32], num_classes=10, width = args.width)

    if args.set == "CIFAR100":
        model = resnet.ResNetWidth18([3, 32, 32], num_classes=100, width = args.width)

    if args.set == "imagenet":
        model = resnet.ResNet50(num_classes=1000)
    
    if args.set == "tiny-imagenet":
        model = resnet.ResNet50(num_classes=200)

    if model is None:
        raise ValueError("Model not found!")
    
    print(f"=> Num model params {sum(p.numel() for p in model.parameters())}")

    # applying sparsity to the network for DNWConv and GMPConv
    if args.conv_type != "DenseConv":

        print(f"==> Setting prune rate of network to {1 - args.init_density}")

        def _sparsity(m):
            if hasattr(m, "set_prune_rate"):
                m.set_prune_rate(1 - args.init_density)

        model.apply(_sparsity)

    # freezing the weights if we are only doing mask training
    if args.freeze_weights:
        print(f"=> Freezing model weights")

        def _freeze(m):
            if hasattr(m, "mask"):
                m.weight.requires_grad = False
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = False

        model.apply(_freeze)

    return model


def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            pass #print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            pass #print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        sparse_thresh = [v for n, v in parameters if ("sparseThreshold" in n) and v.requires_grad]
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        # rest_params = [v for n, v in parameters if ("bn" not in n) and ('sparseThreshold' not in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and ("sparseThreshold" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {
                    "params": sparse_thresh,
                    "weight_decay": args.st_decay if args.st_decay is not None else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer

def sparseInit(model):
    if args.sparse_init == 'uniform':
        model = prune_random_uniform(model)
    elif args.sparse_init == 'ERK':
        model = prune_random_ERK(model)
    elif args.sparse_init == 'balanced':
        model = prune_random_balanced(model)

    return model


def prune_random_uniform(model):
    total_num = 0
    total_den = 0

    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvMask)):
            score = (m.mask.to(m.weight.device) * torch.randn_like(m.weight).to(m.weight.device)).detach().abs_()
            global_scores = torch.flatten(score)
            k = int((1 - args.init_density) * global_scores.numel())
            if k == 0:
                threshold = 0
            else:
                threshold, _ = torch.kthvalue(global_scores, k)
            print(f"Layer: {n}, params: {global_scores.numel()}, threshold: {threshold}")

            score = score.to(m.weight.device)
            zero = torch.tensor([0.0]).to(m.weight.device)
            one = torch.tensor([1.0]).to(m.weight.device)
            m.mask = torch.where(score > threshold, one, zero)  
            total_num += (m.mask == 1).sum()
            total_den += m.mask.numel()
            # m.set_er_mask(args.init_density)
    print('Overall model density after uniform: ', total_num / total_den)
    return model

def prune_random_ERK(model):
    sparsity_list = []
    num_params_list = []
    total_params = 0
    score_list = {}
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvMask)):
            score_list[n] = (m.mask.to(m.weight.device) * torch.randn_like(m.weight).to(m.weight.device)).detach().abs_()
            sparsity_list.append(torch.tensor(m.weight.shape).sum() / m.weight.numel())
            num_params_list.append(m.weight.numel())
            total_params += m.weight.numel()

    num_params_kept = (torch.tensor(sparsity_list) * torch.tensor(num_params_list)).sum()
    num_params_to_keep = total_params * args.init_density
    C = num_params_to_keep / num_params_kept
    sparsity_list = [torch.clamp(C * s, 0, 1) for s in sparsity_list]

    layer = 0
    total_num = 0
    total_den = 0
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvMask)):
            # m.set_er_mask(sparsity_list[layer])
            global_scores = torch.flatten(score_list[n])
            k = int((1 - sparsity_list[layer]) * global_scores.numel())
            if k == 0:
                threshold = 0
            else:
                threshold, _ = torch.kthvalue(global_scores, k)
            print(f"Layer: {layer}, params: {global_scores.numel()}, threshold: {threshold}")

            score = score_list[n].to(m.weight.device)
            zero = torch.tensor([0.0]).to(m.weight.device)
            one = torch.tensor([1.0]).to(m.weight.device)
            m.mask = torch.where(score > threshold, one, zero)
            total_num += (m.mask == 1).sum()
            total_den += m.mask.numel()
            layer += 1

    print('Overall model density after ERK: ', total_num / total_den)
    return model

def prune_random_balanced(model):
    total_params = 0
    layers = 0
    density_list = []
    score_list = {}
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvMask)):
            total_params += m.weight.numel()
            layers += 1

    X = args.init_density * total_params / (layers + 1e-6)

    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvMask)):
            score_list[n] = (m.mask.to(m.weight.device) * torch.randn_like(m.weight).to(m.weight.device)).detach().abs_()
            density_list.append(torch.clamp(torch.tensor(X) / m.weight.numel(), 0, 1))

    total_num = 0
    total_den = 0
    layer = 0
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvMask)):
            global_scores = torch.flatten(score_list[n])
            k = int((1 - density_list[layer]) * global_scores.numel())
            if k == 0:
                threshold = 0
            else:
                threshold, _ = torch.kthvalue(global_scores, k)
            print(f"Layer: {layer}, params: {global_scores.numel()}, threshold: {threshold}")

            score = score_list[n].to(m.weight.device)
            zero = torch.tensor([0.0]).to(m.weight.device)
            one = torch.tensor([1.0]).to(m.weight.device)
            m.mask = torch.where(score > threshold, one, zero)
            # m.mask = torch.zeros_like(m.mask)
            total_num += (m.mask == 1).sum()
            total_den += m.mask.numel()
            layer += 1

    print('Overall model density after Balanced Pruning: ', total_num / total_den)

    return model


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")
    
    prune_rate = 1 - args.init_density

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={prune_rate:.1f}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={prune_rate:.1f}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Base Config, "
            "Name, "
            "Prune Rate, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}\n"
            ).format(now=now, **kwargs)
        )


if __name__ == "__main__":
    main()
