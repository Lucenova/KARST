# -*- coding: utf-8 -*-
import argparse
import copy
import os
import random
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from avalanche.evaluation.metrics.accuracy import Accuracy
from timm.models import create_model
from timm.models.vision_transformer import VisionTransformer, Attention, Mlp
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.utils import AverageMeter, accuracy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.fgfs import FGFS, write, create_transform
from modules.kronecker_adaption import KroneckerAdaption


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args):
    losses_m = AverageMeter()
    model = model.cuda()
    model.train()

    for batch_idx, (input, target) in enumerate(loader):
        input, target = input.cuda(), target.cuda()
        output = model(input)
        loss = loss_fn(output, target)
        losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lrl = [param_group['lr'] for param_group in optimizer.param_groups]
        lr = sum(lrl) / len(lrl)
        write(
            '\rTrain     Epoch: {:>4d}     Iter: {:>4d}/{}     '
            'Loss: {loss.val:#.4g} ({loss.avg:#.4g})  '
            'LR: {lr:.3e}      GPU mem : {mem:.2f} MB'.format(
                epoch,
                batch_idx + 1, len(loader),
                loss=losses_m,
                lr=lr,
                mem=(torch.cuda.max_memory_allocated() / 1024 ** 2)), log_file=args.log_file, end='')

    write('', log_file=args.log_file)


def validate(model, loader):
    top1_m = AverageMeter()

    model.eval()

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1_m.update(acc1.item(), output.size(0))

    write('Test  Samples : {top1.count}    Acc@1: {top1.avg:>7.4f}'.format(top1=top1_m), args.log_file)
    return top1_m


@torch.no_grad()
def test(model: VisionTransformer, test_dataloader: DataLoader):
    model.eval()
    acc = Accuracy()
    model = model.cuda()
    for batch in tqdm(test_dataloader):
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y)
    return acc.result()


def karst_forward_attention(self, x: torch.Tensor):
    batch_size, num_patches, channels = x.shape
    # qkv size is batch_size * num_patches * (3 * 768)
    qkv = self.attention_karst(x)
    qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(batch_size, num_patches, channels)
    x = x * self.st_x_s + self.st_x_b

    proj = self.proj_karst(x)
    proj = proj * self.st_proj_s + self.st_proj_b
    x = self.proj_drop(proj)

    return x


def karst_forward_mlp(self, x: torch.Tensor):
    h = self.fc1_karst(x)
    x = self.act(h)
    x = self.drop1(x)
    h = self.fc2_karst(x)

    h = h * self.st_h_s + self.st_h_b
    x = self.drop2(h)

    return x


def set_karst_module(model: nn.Module, args: Namespace):
    for module in model.children():
        if type(module) == Attention:
            kronecker_w2_a_dims = 768 // args.factor
            module.attention_karst = KroneckerAdaption(
                module.qkv,
                kronecker_w2_a_dims=[kronecker_w2_a_dims, kronecker_w2_a_dims, kronecker_w2_a_dims],
                lora_dim=args.rank,
                factor=args.factor,
                multiplier=args.multiplier,
            )
            module.proj_karst = KroneckerAdaption(
                module.proj,
                lora_dim=args.rank,
                factor=args.factor,
                multiplier=args.multiplier,
            )

            module.st_x_s = nn.Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
            module.st_x_b = nn.Parameter(torch.zeros(1, dtype=torch.float), requires_grad=True)

            module.st_proj_s = nn.Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
            module.st_proj_b = nn.Parameter(torch.zeros(1, dtype=torch.float), requires_grad=True)

            setattr(module, 'forward', karst_forward_attention.__get__(module, module.__class__))
        elif type(module) == Mlp:
            module.fc1_karst = KroneckerAdaption(
                module.fc1,
                lora_dim=args.rank,
                factor=args.factor,
                multiplier=args.multiplier,
            )
            module.fc2_karst = KroneckerAdaption(
                module.fc2,
                lora_dim=args.rank,
                factor=args.factor,
                multiplier=args.multiplier,
            )

            module.st_h_s = nn.Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
            module.st_h_b = nn.Parameter(torch.zeros(1, dtype=torch.float), requires_grad=True)

            setattr(module, 'forward', karst_forward_mlp.__get__(module, module.__class__))
        elif len(list(module.children())) != 0:
            set_karst_module(module, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', default=None, type=str, help='data dir')
    parser.add_argument('--dataset', default='oxford-pets-FS', type=str,
                        choices=['fgvc-aircraft-FS', 'food101-FS', 'oxford-flowers102-FS', 'oxford-pets-FS',
                                 'standford-cars-FS']
                        )

    parser.add_argument('--batch_size', default=32, type=int, help='batch size')

    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--multiplier', default=1, type=float, help='multiplier')
    parser.add_argument('--factor', default=8, type=int, help='factor')
    parser.add_argument('--rank', default=8, type=int, help='rank')

    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--model', default='vit_base_patch16_224_in21k', type=str)
    parser.add_argument('--save_dir', default='models/few_shot_karst/', type=str)

    parser.add_argument('--shot', default=1, type=int)

    args = parser.parse_args()

    if args.dataset == 'fgvc-aircraft-FS':
        args.num_classes = 100
    elif args.dataset == 'food101-FS':
        args.num_classes = 101
    elif args.dataset == 'oxford-flowers102-FS':
        args.num_classes = 102
    elif args.dataset == 'oxford-pets-FS':
        args.num_classes = 37
    elif args.dataset == 'standford-cars-FS':
        args.num_classes = 196
    else:
        raise NotImplementedError

    benchmark = 'FGFS'
    dataset_func = FGFS
    train_transform_type = 'FGFS_train'
    test_transform_type = 'FGFS_test'
    val_split = 'val'
    test_split = 'test'

    args.log_dir = os.path.join(
        'checkpoint',
        args.model,
        'karst',
        benchmark,
        args.dataset,
        f'wd_{args.wd}_lr_{args.lr}_multiplier_{args.multiplier}_factor_{args.factor}_rank_{args.rank}'
    )

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if benchmark == 'FGFS':
        args.log_file = os.path.join(args.log_dir, 'log_shot_{}.txt'.format(args.shot))
    else:
        raise NotImplementedError

    if os.path.isfile(args.log_file):
        os.remove(args.log_file)

    write(args, args.log_file)
    set_seed(args.seed)

    checkpoint_path = ''
    drop_path_rate = 0.1

    accs = []
    for fseed in range(3):
        train_split = 'train_shot_{}_seed_{}'.format(args.shot, fseed)
        vit = create_model(args.model, checkpoint_path=checkpoint_path, drop_path_rate=drop_path_rate)
        vit.reset_classifier(args.num_classes)

        set_karst_module(vit, args)

        trainable = []
        total_param = 0
        for n, p in vit.named_parameters():
            if 'karst' in n or 'st' in n or 'head' in n:
                trainable.append(p)
                p.requires_grad = True
                if 'head' not in n:
                    total_param += p.numel()
                # print('current trainable params: ', n, p.numel(), p.requires_grad)
            elif 'norm' in n:
                p.requires_grad = True
                trainable.append(p)
                total_param += p.numel()
                # print('current trainable params: ', n, p.numel(), p.requires_grad)
            else:
                p.requires_grad = False

        print('total trainable params: ', total_param)
        optimizer = AdamW(trainable, lr=args.lr, weight_decay=float(args.lr) / 10)
        lr_scheduler = CosineLRScheduler(
            optimizer, t_initial=args.epochs, warmup_t=10, lr_min=float(args.lr) / 10, warmup_lr_init=(args.lr) / 100
        )
        args.epochs = lr_scheduler.get_cycle_length() + 10

        # create the train and eval datasets
        dataset_train = dataset_func(
            root=args.data_dir, dataset=args.dataset, split_=train_split,
            transform=create_transform(aug_type=train_transform_type), log_file=args.log_file)
        dataset_val = dataset_func(
            root=args.data_dir, dataset=args.dataset, split_=val_split,
            transform=create_transform(aug_type=test_transform_type),
            log_file=args.log_file
        )

        dataset_test = dataset_func(
            root=args.data_dir,
            dataset=args.dataset,
            split_=test_split,
            transform=create_transform(aug_type=test_transform_type), log_file=args.log_file
        )

        write('len of train_set : {} train_transform : {}'.format(len(dataset_train), dataset_train.transform),
              args.log_file)
        write('len of val_set : {} val_transform : {}'.format(len(dataset_val), dataset_val.transform),
              args.log_file)
        write('len of test_set : {} eval_transform : {}'.format(len(dataset_test), dataset_test.transform),
              args.log_file)

        loader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )

        loader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size * 4,
            shuffle=False,
            num_workers=4,
        )

        loader_test = DataLoader(
            dataset_test,
            batch_size=args.batch_size * 4,
            shuffle=False,
            num_workers=4,
        )

        write('Using CrossEntropyLoss', args.log_file)
        criterion = nn.CrossEntropyLoss()
        write('Training in FP32', args.log_file)

        best_val_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            train_one_epoch(epoch, vit, loader_train, optimizer, criterion, args)
            lr_scheduler.step(epoch)

            if epoch % 10 == 0:
                top1_acc_val = validate(vit, loader_val)
                val_acc = top1_acc_val.avg

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_dict = copy.deepcopy(vit.state_dict())

        vit.load_state_dict(best_dict)
        top1_acc_final_test = validate(vit, loader_test)
        write('fseed : {}  epoch: {}  eval_acc: {:.2f}'.format(fseed, epoch, top1_acc_final_test.avg),
              log_file=args.log_file)
        accs.append(top1_acc_final_test.avg)

    write('Overall Mean Acc with {} fseeds : {:.2f}'.format(len(accs), np.mean(accs)), args.log_file)