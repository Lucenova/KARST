# -*- coding: utf-8 -*-
"""
# @project    : KARST
# @author     : https://github.com/Lucenova
# @date       : 2025/2/10 10:57
# @brief      : 
"""
import os
import random
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import torch.nn as nn
import yaml
from avalanche.evaluation.metrics.accuracy import Accuracy
from timm.models import create_model
from timm.models.vision_transformer import VisionTransformer, Attention, Mlp
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vtab import get_data, get_classes_num
from modules.kronecker_adaption import KroneckerAdaption

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
    qkv = self.attention_lokr(x)
    qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(batch_size, num_patches, channels)
    x = x * self.st_x_s + self.st_x_b

    proj = self.proj_lokr(x)
    proj = proj * self.st_proj_s + self.st_proj_b
    x = self.proj_drop(proj)

    return x

def karst_forward_mlp(self, x: torch.Tensor):
    h = self.fc1_ka(x)
    x = self.act(h)
    x = self.drop1(x)
    h = self.fc2_ka(x)

    h = h * self.st_h_s + self.st_h_b
    x = self.drop2(h)

    return x

def set_karst_module(model: nn.Module, args: Namespace):
    for module in model.children():
        if type(module) == Attention:
            kronecker_w2_a_dims = 768 // args.factor
            module.attention_ka = KroneckerAdaption(
                module.qkv,
                kronecker_w2_a_dims=[kronecker_w2_a_dims, kronecker_w2_a_dims, kronecker_w2_a_dims],
                lora_dim=args.rank,
                factor=args.factor,
                multiplier=args.multiplier,
                group_num=args.group_num,
            )
            module.proj_ka = KroneckerAdaption(
                module.proj,
                lora_dim=args.rank,
                factor=args.factor,
                multiplier=args.multiplier,
                group_num=args.group_num,
            )

            module.st_x_s = nn.Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
            module.st_x_b = nn.Parameter(torch.zeros(1, dtype=torch.float), requires_grad=True)

            module.st_proj_s = nn.Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
            module.st_proj_b = nn.Parameter(torch.zeros(1, dtype=torch.float), requires_grad=True)

            setattr(module, 'forward', karst_forward_attention.__get__(module, module.__class__))
        elif type(module) == Mlp:
            module.fc1_ka = KroneckerAdaption(
                module.fc1,
                lora_dim=args.rank,
                factor=args.factor,
                multiplier=args.multiplier,
                group_num=args.group_num,
            )
            module.fc2_ka = KroneckerAdaption(
                module.fc2,
                lora_dim=args.rank,
                factor=args.factor,
                multiplier=args.multiplier,
                group_num=args.group_num,
            )

            module.st_h_s = nn.Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
            module.st_h_b = nn.Parameter(torch.zeros(1, dtype=torch.float), requires_grad=True)

            setattr(module, 'forward', karst_forward_mlp.__get__(module, module.__class__))
        elif len(list(module.children())) != 0:
            set_karst_module(module, args)

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def save(args: Namespace, model: VisionTransformer, acc: float, epoch: int):
    model.eval()
    model.cpu()
    trainable = {}
    for n, p in model.named_parameters():
        if 'ka' in n or 'st' in n or 'head' in n:
            trainable[n] = p.data

        if args.layer_norm and 'norm' in n:
            trainable[n] = p.data

    save_model_path = args.save_dir
    if not os.path.exists(f'{save_model_path}{args.dataset}'):
        os.makedirs(f'{save_model_path}{args.dataset}')

    save_file = f'{save_model_path}{args.dataset}/multiplier_{args.multiplier}_group_{args.group_num}_lr_{args.lr}_rank_{args.rank}_drop_{args.drop_path}.log'
    with open(save_file, 'w') as f:
        f.write(str(epoch) + ' ' + str(acc))

def get_config(dataset_name):
    with open('./configs/%s.yaml' % (dataset_name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log', action='store_true')

    parser.add_argument('--multiplier', type=float, default=1)
    parser.add_argument('--factor', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--drop_path', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='cifar')

    parser.add_argument('--layer_norm', action='store_true')
    parser.add_argument('--save_dir', type=str, default='models/kfc_vit_vtab_drop_multi_group/')

    parser.add_argument('--group_num', type=int, default=8)
    args = parser.parse_args()

    seed = args.seed
    dataset = args.dataset
    model = args.model

    best_acc = 0
    checkpoint_path = ''

    set_seed(seed)
    train_dataloader, test_dataloader = get_data(dataset, batch_size=32)

    print('args: ', args)
    vit = create_model(args.model, checkpoint_path=checkpoint_path, drop_path_rate=args.drop_path)
    vit.reset_classifier(get_classes_num(dataset))
    set_karst_module(vit, args)

    trainable = []
    total_param = 0
    for n, p in vit.named_parameters():
        if 'ka' in n or 'st' in n or 'head' in n:
            trainable.append(p)
            p.requires_grad = True
            if 'head' not in n:
                total_param += p.numel()
                print('current trainable params: ', n, p.numel(), p.requires_grad)
        elif args.layer_norm and 'norm' in n:
            p.requires_grad = True
            trainable.append(p)
            total_param += p.numel()
            print('current trainable params: ', n, p.numel(), p.requires_grad)
        else:
            p.requires_grad = False

    print('total trainable params: ', total_param)
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=float(args.lr) / 10)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        cycle_decay=0.5,
        lr_min=float(args.lr) / 10,
        warmup_lr_init=float(args.lr) / 100,
        warmup_t=10
    )

    total_epoch = scheduler.get_cycle_length() + 10

    model = vit.cuda()
    model.train()
    pbar = tqdm(range(total_epoch))
    for ep in pbar:
        for i, batch in enumerate(train_dataloader):
            model = model.cuda()
            model.train()
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = F.cross_entropy(out, y)
            print('loss:', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step(ep)
        if torch.cuda.is_available():
            MB = 1024.0 * 1024.0
            peak_memory = (torch.cuda.max_memory_allocated() / MB) / 1000
            print("Epoch Memory utilization: %.3f GB" % peak_memory)
        if ep % 10 == 9:
            acc = test(vit, test_dataloader)
            if acc > best_acc:
                best_acc = acc
                save(args, model, acc, ep)
            pbar.set_description(str(acc) + '|' + str(best_acc))

    model = model.cpu()
    print('acc1:', best_acc)
