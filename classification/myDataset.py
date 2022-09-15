import os
import torch 
import torchvision.transforms as transforms 
import torchvision 
import random

import torch

from typing import Dict
import pandas as pd
from PIL import Image
import torchvision
import torch


class ImageIterableDatasetMultiTargetWithDynLabels(torch.utils.data.IterableDataset):
    
    def __init__(
        self, 
        path:str,
        target_mapping: Dict[str, int],
        meta_path=None,
        lat_key = "LAT", 
        lon_key = "LON",
        transformation = None,
        shuffle=True,
    ):

        super(ImageIterableDatasetMultiTargetWithDynLabels, self).__init__()

        self.target_mapping = target_mapping
        self.transformation = transformation 
        self.seed = random.randint(1, 100)
        self.shuffle = shuffle
        self.path = path # images dir
        self.meta_path = meta_path # {...}_365.csv file containing information about LAT, LON
        self.meta = None # pandas of the meta info.
        self.lat_key = lat_key
        self.lon_key = lon_key

        if not isinstance(self.path, (list, set)):
            self.path = [self.path]
        
        if meta_path is not None:
            self.meta = pd.read_csv(meta_path, index_col=0)
            self.meta = self.meta.astype({lat_key: "float32", lon_key:"float32"})

        self.length = len(self.target_mapping) 

        self.images = list(target_mapping.keys())
        

    def __len__(self):
        return self.length
    

    def __iter__(self):

        images_indices = list(range(self.length))
        images_files = list(self.target_mapping.keys())

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(images_indices)

        # parallelizing the process
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            def split_list(alist, splits=1):
                length = len(alist)
                return [
                    alist[i*length//splits:(i+1)*length // splits]
                    for i in range(splits)
                ]

            images_indices_split = split_list(images_indices, worker_info.num_workers)[worker_info.id]
        else:
            images_indices_split = images_indices

        cache = []
        
        for idx in images_indices_split:
            
            _id = images_files[idx]
            img = Image.open(os.path.join(self.path[0], _id))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if img.width > 320 and img.height > 320:
                img = torchvision.transforms.Resize(320)(img)
            
            img = self.transformation(img)

            if len(self.target_mapping[_id]) == 1:
                target = self.target_mapping[_id][0]
            else:
                target = self.target_mapping[_id]


            if self.meta_path is None:
                yield img, target 
            else:
                meta = self.meta.loc[_id]
                yield img, target, meta[self.lat_key], meta[self.lon_key]
                