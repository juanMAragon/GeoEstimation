import os 
import sys
import re 
from typing import Dict
from io import BytesIO
from PIL import Image 
import torch 
import msgpack
import numpy as np
from tqdm import tqdm


dest_path = os.path.join(os.sep, 'media', 'vid1', 'all_jaragon', 'yfcc25600')
# dest_path = os.path.join(os.sep, 'media', 'vid1', 'all_jaragon', 'mp16')
key_img_id = b'id'


for i in tqdm(np.arange(865)):
    # path = os.path.join(os.sep, 'media', 'vid1', 'all_jaragon', 'mp16', f'shard_{i}.msg')
    path = os.path.join(os.sep, 'media', 'vid1', 'all_jaragon', 'yfcc25600', f'shard_{i}.msg')
    if os.path.exists(path):

        with open(path, "rb") as f:
            unpacker = msgpack.Unpacker(f, max_buffer_size=1024*1024*1024, raw=True)
            for i,x in tqdm(enumerate(unpacker), leave=False):
                img_dir = x[key_img_id].decode('utf-8')
                img_dir_folder = os.path.join(dest_path, *img_dir.split(os.sep)[:-1])
                
                if not os.path.exists(img_dir_folder):
                    os.makedirs(img_dir_folder)
                
                img = Image.open(BytesIO(x[b'image']))
                img_dir = os.path.join(dest_path, *img_dir.split(os.sep))
                img.save(img_dir)