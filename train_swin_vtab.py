# -*- coding: utf-8 -*-

import random
from argparse import ArgumentParser, Namespace
from typing import Optional
import yaml
import numpy as np
import torch.nn as nn
import wandb
from avalanche.evaluation.metrics.accuracy import Accuracy
from timm.models import create_model
from timm.models.swin_transformer import SwinTransformer, WindowAttention, Mlp
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vtab import *
from modules.kronecker_adaption import KroneckerAdaption

@torch.no_grad()
def test(model: SwinTransformer, test_dataloader: DataLoader):
    model.eval()
    acc = Accuracy()
    model = model.cuda()
    for batch in tqdm(test_dataloader):
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y)

    return acc.result()


def karst_forward_attention(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
    batch_size, num_patches, channels = x.shape
    # qkv size is batch_size * num_patches * (3 * 768)
    qkv = self.attention_ka(x)
    qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    if self.fused_attn:
        attn_mask = self._get_rel_pos_bias()
        if mask is not None:
            num_win = mask.shape[0]
            mask = mask.view(
                1, num_win, 1, num_patches, num_patches).expand(batch_size // num_win, -1, self.num_heads, -1, -1)
            attn_mask = attn_mask + mask.reshape(-1, self.num_heads, num_patches, num_patches)

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + self._get_rel_pos_bias()
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, num_patches, num_patches) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, num_patches, num_patches)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(batch_size, num_patches, -1)
    x = x * self.st_x_s + self.st_x_b

    proj = self.proj_ka(x)
    proj = proj * self.st_proj_s + self.st_proj_b
    x = self.proj_drop(proj)

    return x


def set_karst_mlp(self, x: torch.Tensor):
    h = self.fc1_ka(x)
    x = self.act(h)
    x = self.drop1(x)
    h = self.fc2_kar(x)

    h = h * self.st_h_s + self.st_h_b
    x = self.drop2(h)

    return x

HIERARCHY_RANK = {
    16: 2,
    32: 2,
    64: 4,
    128: 8,
}


def set_karst_modulue(model: nn.Module, args: Namespace):
    for module in model.children():
        if type(module) == WindowAttention:
            kronecker_w2_a_dims = min(module.qkv.in_features // args.factor, module.qkv.out_features // args.factor)
            module.attention_ka = KroneckerAdaption(
                module.qkv,
                kronecker_w2_a_dims=[kronecker_w2_a_dims, kronecker_w2_a_dims, kronecker_w2_a_dims],
                lora_dim=HIERARCHY_RANK[kronecker_w2_a_dims],
                factor=args.factor,
                multiplier=args.multiplier,
            )
            proj_ka_dim = min(module.proj.in_features // args.factor, module.proj.out_features // args.factor)
            module.proj_ka = KroneckerAdaption(
                module.proj,
                lora_dim=HIERARCHY_RANK[proj_ka_dim],
                factor=args.factor,
                multiplier=args.multiplier,
            )

            module.st_x_s = nn.Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
            module.st_x_b = nn.Parameter(torch.zeros(1, dtype=torch.float), requires_grad=True)

            module.st_proj_s = nn.Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
            module.st_proj_b = nn.Parameter(torch.zeros(1, dtype=torch.float), requires_grad=True)

            setattr(module, 'forward', karst_forward_attention.__get__(module, module.__class__))
        elif type(module) == Mlp:
            fc1_ka_dim = min(module.fc1.in_features // args.factor, module.fc1.out_features // args.factor)
            fc2_kar_dim = min(module.fc2.in_features // args.factor, module.fc2.out_features // args.factor)
            module.fc1_ka = KroneckerAdaption(
                module.fc1,
                lora_dim=HIERARCHY_RANK[fc1_ka_dim],
                factor=args.factor,
                multiplier=args.multiplier,
            )
            module.fc2_kar = KroneckerAdaption(
                module.fc2,
                lora_dim=HIERARCHY_RANK[fc2_kar_dim],
                factor=args.factor,
                multiplier=args.multiplier,
            )

            module.st_h_s = nn.Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
            module.st_h_b = nn.Parameter(torch.zeros(1, dtype=torch.float), requires_grad=True)

            setattr(module, 'forward', set_karst_mlp.__get__(module, module.__class__))
        elif len(list(module.children())) != 0:
            set_karst_modulue(module, args)


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def save(args: Namespace, model: SwinTransformer, acc: float, epoch: int):
    model.eval()
    model.cpu()
    # trainable = {}
    # for n, p in model.named_parameters():
    #     if 'lokr' in n or 'head' in n:
    #         trainable[n] = p.data
    #
    #     if args.layer_norm and 'norm' in n:
    #         trainable[n] = p.data

    save_model_path = args.save_dir
    if not os.path.exists(f'{save_model_path}{args.dataset}'):
        os.makedirs(f'{save_model_path}{args.dataset}')

    # torch.save(trainable, f'{save_model_path}{dataset}_multiplier_{args.multiplier}_factor_{args.factor}_lr_{args.lr}.pth')
    save_file = f'{save_model_path}{args.dataset}/multiplier_{args.multiplier}_lr_{args.lr}_dp_{args.drop_path}.log'
    with open(save_file, 'w') as f:
        f.write(str(epoch) + ' ' + str(acc))


def get_config(dataset_name):
    with open('./configs/swin_karst/%s.yaml' % (dataset_name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log', action='store_true')

    parser.add_argument('--multiplier', type=float, default=1)
    parser.add_argument('--factor', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='swin_base_patch4_window7_224')
    parser.add_argument('--dataset', type=str, default='cifar')

    parser.add_argument('--layer_norm', action='store_true')
    parser.add_argument('--save_dir', type=str, default='models/swin_kfc/')
    parser.add_argument('--drop_path', type=float, default=0.1)
    args = parser.parse_args()

    seed = args.seed
    dataset = args.dataset
    model = args.model

    config = get_config(args.dataset)
    args.lr = float(config['learning_rate'])
    args.multiplier = float(config['multiplier'])
    print('args: ', args)

    best_acc = 0
    vit = create_model(
        'swin_base_patch4_window7_224_in22k',
        drop_path_rate=args.drop_path,
        pretrained=True,
        cache_dir='',
    )

    set_seed(seed)
    train_dataloader, test_dataloader = get_data(dataset, batch_size=32)
    vit.reset_classifier(get_classes_num(dataset))
    set_karst_modulue(vit, args)

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
        warmup_t=10,
        lr_min=float(args.lr) / 10,
        warmup_lr_init=float(args.lr) / 100
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
            if args.log:
                wandb.log({'loss': loss.item()})
            else:
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
            if args.log:
                wandb.log({'acc': acc})
            if acc > best_acc:
                best_acc = acc
                save(args, model, acc, ep)
            pbar.set_description(str(acc) + '|' + str(best_acc))

    print('acc1:', best_acc)