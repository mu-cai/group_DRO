import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.celebA_dataset import CelebADataset
from data.cub_dataset import CUBDataset
from data.dro_dataset import DRODataset
from data.bird_dataset import birdDataset
from data.multinli_dataset import MultiNLIDataset

################
### SETTINGS ###
################

confounder_settings = {
    'CelebA':{
        'constructor': CelebADataset # used 
    },
    'CUB':{
        'constructor': CUBDataset
    },
    'bird':{
        'constructor': birdDataset # used 
    },
    'MultiNLI':{
        'constructor': MultiNLIDataset
    },
    'Waterbird':{
        'constructor': MultiNLIDataset
    }
}

########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(args, train, return_full_dataset=False):
    full_dataset = confounder_settings[args.dataset]['constructor'](
        root_dir=args.root_dir,
        target_name=args.target_name,
        confounder_names=args.confounder_names,
        model_type=args.model,
        augment_data=args.augment_data)
    if return_full_dataset: # never used
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str)
    if train:
        splits = ['train', 'val', 'test'] # used
    else:
        splits = ['test']
    subsets = full_dataset.get_splits(splits, train_frac=args.fraction)

    # if args.ood:
    #     dro_subsets = [ 1]
    #     dro_subsets.extend([DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
    #                         n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str, args=args) \
    #             for split in ['val', 'test']] )

    # else:
    dro_subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
                        n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str, args=args) \
            for split in splits]
    return dro_subsets
