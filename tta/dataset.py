import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset.random_erasing import RandomErasing


import os
import random
from random import randint, shuffle
from random import random as rand
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from dataset.re_dataset import re_train_dataset, re_test_dataset, re_test_dataset_icfg, re_train_dataset_attr, re_test_dataset_attr
from dataset.randaugment import RandomAugment
from dataset.random_erasing import RandomErasing


def create_dataset(dataset, config, evaluate=False, tta=False):
    # gene
    gene_norm = transforms.Normalize((0.4416847, 0.41812873, 0.4237452), (0.3088255, 0.29743394, 0.301009))
    # cuhk
    cuhk_norm = transforms.Normalize((0.38901278, 0.3651612, 0.34836376), (0.24344306, 0.23738699, 0.23368555))
    # icfg
    icfg_norm = transforms.Normalize((0.30941582, 0.28956893, 0.30347288), (0.25849792, 0.24547698, 0.2366199))
    # rstp
    rstp_norm = transforms.Normalize((0.27722597, 0.26065794, 0.3036557), (0.2609547, 0.2508087, 0.25293276))
    # # pa100k
    # pa100k_norm = transforms.Normalize((0.46485138, 0.45038012, 0.4632019), (0.25088054, 0.24609283, 0.24240193))

    if dataset == 're_cuhk':
        train_norm = cuhk_norm
        test_norm = cuhk_norm
    elif dataset == 're_icfg':
        train_norm = icfg_norm
        test_norm = icfg_norm
    elif dataset == 're_rstp':
        train_norm = rstp_norm
        test_norm = rstp_norm
    elif dataset == 're_gene':
        train_norm = gene_norm
        test_norm = cuhk_norm
    # elif dataset == 're_pa100k':
    #     train_norm = pa100k_norm
    #     test_norm = pa100k_norm

    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop((config['h'], config['h']),
        #                              scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize',
                                              'Brightness', 'Sharpness', 'ShearX',
                                              'ShearY', 'TranslateX', 'TranslateY',
                                              'Rotate']),
        transforms.ToTensor(),
        train_norm,
        RandomErasing(probability=config['erasing_p'], mean=[0.0, 0.0, 0.0])
    ])

    pre_transform = transforms.Compose([
        transforms.RandomResizedCrop((config['h'], config['h']),
                                     scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize',
                                              'Brightness', 'Sharpness', 'ShearX',
                                              'ShearY', 'TranslateX', 'TranslateY',
                                              'Rotate']),
        transforms.ToTensor(),
        train_norm,
        RandomErasing(probability=config['erasing_p'], mean=[0.0, 0.0, 0.0])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        test_norm,
    ])

    if dataset == 're_icfg':
        test_dataset = re_test_dataset_icfg(config, test_transform)
        if tta:
            return None, None, test_dataset
        if evaluate:
            return None, test_dataset
        train_dataset = re_train_dataset(config, train_transform, pre_transform)
        return train_dataset, test_dataset
    # elif dataset == 're_pa100k':
    #     test_dataset = re_test_dataset_attr(config['test_file'], config, test_transform)
    #     if tta:
    #         return None, None, test_dataset
    #     val_dataset = re_test_dataset_attr(config['val_file'], config, test_transform)
    #     if evaluate:
    #         return None, val_dataset, test_dataset
    #     train_dataset = re_train_dataset_attr(config, train_transform)
    #     return train_dataset, val_dataset, test_dataset
    else:
        test_dataset = re_test_dataset(config['test_file'], config, test_transform)
        if tta:
            return None, None, test_dataset
        val_dataset = re_test_dataset(config['val_file'], config, test_transform)
        if evaluate:
            return None, val_dataset, test_dataset
        train_dataset = re_train_dataset(config, train_transform, pre_transform)
        return train_dataset, val_dataset, test_dataset


def create_loader(datasets, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, bs, n_worker, is_train, collate_fn in zip(datasets, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = True
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders



class search_tta_dataset(Dataset):
    def __init__(self, config, tta_transform, sims_matrix_t2i, image_embeds, text_embeds, text_atts, recall_types, ss_idxs_list, uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list):
        # ann_file = config['tta_file']
        # self.transform = transform
        # self.image_root = config.get('image_root_tta', config['image_root'])
        # self.max_words = config['max_words']# 56

        # self.ann = read_json_to_list(ann_file)

        # self.be_pose_img = config.get('be_pose_img', False)
        # print('tta dataset -->    be_pose_img:', self.be_pose_img)

        # self.text = []
        # self.image = []
        # self.g_pids = []
        # self.q_pids = []
        # for img_id, ann in enumerate(self.ann):
        #     self.g_pids.append(ann['image_id'])
        #     self.image.append(ann['image'])
        #     for i, caption in enumerate(ann['caption']):
        #         self.q_pids.append(ann['image_id'])
        #         self.text.append(pre_caption(caption, self.max_words))
        self.config = config
        # self.transform = tta_transform
        self.sims_matrix_t2i = sims_matrix_t2i
        self.image_embeds= image_embeds
        self.text_embeds = text_embeds
        self.text_atts = text_atts
        # self.recall_types = recall_types
        # self.ss_idxs_list = ss_idxs_list
        self.uncertaintys_list = uncertaintys_list
        # if config.get('uncertainty_temper_is_learnable', False) == True:
        if 1:
            self.proba_top1_sim_list = proba_top1_sim_list
            self.proba_inversed_sim_list = proba_inversed_sim_list

        if config.get('sample_selection', 'all') == 'top1':
            self.sims_matrix_t2i = sims_matrix_t2i[ss_idxs_list]
            #### self.image_embeds= image_embeds[ss_idxs_list]
            self.text_embeds = text_embeds[ss_idxs_list]
            self.text_atts = text_atts[ss_idxs_list]
            # self.labels = [labels[i] for i in ss_idxs_list]
            # self.recall_types = [recall_types[i] for i in ss_idxs_list]
            self.uncertaintys_list = [uncertaintys_list[i] for i in ss_idxs_list]
            # if config.get('uncertainty_temper_is_learnable', False) == True:
            if 1:
                self.proba_top1_sim_list = [proba_top1_sim_list[i] for i in ss_idxs_list]
                self.proba_inversed_sim_list = [proba_inversed_sim_list[i] for i in ss_idxs_list]

    def __len__(self):
        return len(self.sims_matrix_t2i)

    def __getitem__(self, index):
        # image_path = os.path.join(self.image_root, self.ann[index]['image'])
        # image = Image.open(image_path).convert('RGB')
        # image = self.transform(image)

        # if self.be_pose_img:
        #     pose_path = os.path.join(self.image_root, 'pose/' + self.ann[index]['image'])
        #     pose = Image.open(pose_path).convert('RGB')
        #     pose = self.transform(pose)
        # else:
        #     pose = {}

        # return image, pose, index
        topk_sim, topk_idx = self.sims_matrix_t2i[index].topk(k=self.config['k_tta'], dim=0) #[k_tta]
        encoder_output = self.image_embeds[topk_idx] #[k_tta, 50, 1024]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long) #[k_tta, 50])
        text_embeds = self.text_embeds[index].repeat(self.config['k_tta'], 1, 1) #k_tta, 56, 768])
        text_atts = self.text_atts[index].repeat(self.config['k_tta'], 1) #k_tta, 56
        uncertainty = self.uncertaintys_list[index]
        proba_top1_sim = self.proba_top1_sim_list[index]
        proba_inversed_sim = self.proba_inversed_sim_list[index]

        return encoder_output, encoder_att, text_embeds, text_atts, uncertainty, proba_top1_sim, proba_inversed_sim


def create_tta_dataset(config, sims_matrix_t2i, image_embeds, text_embeds, text_atts, recall_types, ss_idxs_list, uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list,):

    tta_dataset = search_tta_dataset(config, None, sims_matrix_t2i, image_embeds, text_embeds, text_atts, recall_types, ss_idxs_list, uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list)

    return tta_dataset


def create_tta_loader(datasets, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, bs, n_worker, is_train, collate_fn in zip(datasets, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = True
            drop_last = True
        else:
            shuffle = False
            drop_last = False

        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders



# class search_tta_img_aug_dataset(Dataset):
#     def __init__(self, config, transform):
#         ann_file = config['tta_file']
#         self.transform = transform
#         self.image_root = config.get('image_root_tta', config['image_root'])
#         self.max_words = config['max_words']

#         self.ann = read_json_to_list(ann_file)

#         self.be_pose_img = config.get('be_pose_img', False)
#         print('     tta img aug dataset -->    be_pose_img:', self.be_pose_img)

#         self.text = []
#         self.image = []
#         self.g_pids = []
#         self.q_pids = []
#         for img_id, ann in enumerate(self.ann):
#             self.g_pids.append(ann['image_id'])
#             self.image.append(ann['image'])
#             for i, caption in enumerate(ann['caption']):
#                 self.q_pids.append(ann['image_id'])
#                 self.text.append(pre_caption(caption, self.max_words))
#         pass

#     def __len__(self):
#         return len(self.image)

#     def __getitem__(self, index):
#         image_path = os.path.join(self.image_root, self.ann[index]['image'])
#         image = Image.open(image_path).convert('RGB')
#         image = self.transform(image)

#         if self.be_pose_img:
#             pose_path = os.path.join(self.image_root, 'pose/' + self.ann[index]['image'])
#             pose = Image.open(pose_path).convert('RGB')
#             pose = self.transform(pose)
#         else:
#             pose = {}

#         return image, pose, index


# def create_tta_img_aug_dataset(config):

#     normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

#     tta_img_aug_transform = transforms.Compose([
#         transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#         RandomErasing(probability=config['erasing_p'], mean=[0.0, 0.0, 0.0])
#     ])

#     tta_img_aug_dataset = search_tta_img_aug_dataset(config, tta_img_aug_transform)

#     return tta_img_aug_dataset


# def create_tta_img_aug_loader(datasets, batch_size, num_workers, is_trains, collate_fns):
#     loaders = []
#     for dataset, bs, n_worker, is_train, collate_fn in zip(datasets, batch_size, num_workers, is_trains, collate_fns):
#         if is_train:
#             shuffle = True
#             drop_last = False
#         else:
#             shuffle = False
#             drop_last = False

#         loader = DataLoader(
#             dataset,
#             batch_size=bs,
#             num_workers=n_worker,
#             pin_memory=True,
#             shuffle=shuffle,
#             collate_fn=collate_fn,
#             drop_last=drop_last,
#         )
#         loaders.append(loader)

#     if len(loaders) <= 1:
#         print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

#     return loaders