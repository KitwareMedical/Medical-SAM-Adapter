import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_click


class ChestXRay(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click',
                 plane=False):
        """data_path might be M:\Dev\CXR\LungAI\Data"""

        if mode == 'Training':
            df = pd.read_csv(os.path.join(data_path, 'PreprocessedData-YOLO/train.txt'), header=None)
        elif mode == 'Test':
            df = pd.read_csv(os.path.join(data_path, 'PreprocessedData-YOLO/test.txt'), header=None)

        children_list = df[df[0].str.contains('Children')]
        shenzhen_list = df[df[0].str.contains('Shenzhen')]
        montgomery_list = df[df[0].str.contains('Montgomery')]
        children_size = len(children_list)

        if mode == 'Training':
            # balance the dataset (Children's is the smallest)
            # df = pd.concat([children_list, shenzhen_list.head(children_size), montgomery_list.head(children_size)])
            df = pd.concat([shenzhen_list, montgomery_list])
        elif mode == 'Test':
            # for evaluation, we only care about the performance on Children's dataset
            # df = children_list
            df = pd.concat([shenzhen_list, montgomery_list])

        df = df.sample(frac=1, random_state=1983)  # shuffle the dataframe

        self.name_list = list(df[0])

        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, 'ReorganizedImage', name + '.png')
        msk_path = os.path.join(self.data_path, 'ReorganizedSegmentation', name + '_label.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)

            if self.transform_msk:
                mask = self.transform_msk(mask).int()

            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        # name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}
        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
        }
