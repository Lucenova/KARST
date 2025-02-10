# -*- coding: utf-8 -*-
"""
# @project    : PETL-ViT
# @author     : https://github.com/zhuyuedlut
# @date       : 2024/5/14 09:30
# @brief      : 
"""
import os

from timm.data.auto_augment import rand_augment_transform
from timm.data.transforms_factory import RandomResizedCropAndInterpolation
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.transforms.functional import InterpolationMode


def create_transform(aug_type=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    if aug_type == 'VTAB':
        tfl = [
            transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]

    elif aug_type == 'FGVC_train':
        tfl = [
            RandomResizedCropAndInterpolation(224, interpolation='bicubic'),
            transforms.RandomHorizontalFlip(),
        ]

    elif aug_type == 'FGVC_test':

        tfl = [
            transforms.Resize(248, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]

    elif aug_type == 'FGFS_train':

        tfl = [
            RandomResizedCropAndInterpolation(224, interpolation='bicubic'),
            transforms.RandomHorizontalFlip(),
        ]

        img_size_min = 224
        aa_params = dict(translate_const=int(img_size_min * 0.45),
                         img_mean=tuple([min(255, round(255 * x)) for x in mean]), interpolation=3)
        tfl += [rand_augment_transform('rand-m9-mstd0.5-inc1', aa_params)]

    elif aug_type == 'FGFS_test':

        tfl = [
            transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
    elif aug_type == 'efficientnet_test':
        tfl = [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
    else:
        raise NotImplementedError

    tfl += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    transform = transforms.Compose(tfl)

    return transform


def write(print_obj, log_file=None, end='\n'):
    print(print_obj, end=end)
    if log_file is not None:
        with open(log_file, 'a') as f:
            print(print_obj, end=end, file=f)


class FGFS(ImageFolder):
    def __init__(self, root, dataset, split_, transform, log_file=None):
        self.root = root
        self.dataset = dataset.replace('-FS', '')

        self.loader = default_loader
        self.target_transform = None
        self.transform = transform

        if 'train' in split_:
            shot = split_.split('_')[2]
            seed = split_.split('_')[4]
            list_path = os.path.join(
                self.root, 'few-shot_split', self.dataset,
                'annotations/train_meta.list.num_shot_' + shot + '.seed_' + seed)
        elif 'val' in split_:
            list_path = os.path.join(self.root, 'few-shot_split', self.dataset, 'annotations/val_meta.list')
        elif 'test' in split_:
            list_path = os.path.join(self.root, 'few-shot_split', self.dataset, 'annotations/test_meta.list')
        else:
            raise NotImplementedError

        write('list_path : {}'.format(list_path), log_file=log_file)

        self.samples = []
        with open(list_path, 'r') as f:
            for line in f:
                img_name = line.rsplit(' ', 1)[0]
                label = int(line.rsplit(' ', 1)[1])
                self.samples.append((os.path.join(root, self.dataset, img_name), label))
