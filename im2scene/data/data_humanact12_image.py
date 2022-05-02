"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
Heavily modified by James Mullen
"""

import os
import tqdm
import pickle
import numpy as np
import torch.utils.data
from PIL import Image


class VideoFolderDataset(torch.utils.data.Dataset):
    # Uses HumanAct12 Folder
    def __init__(self, folder, cache, min_len=32):
        self.lengths = []
        self.videos = []

        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.videos, self.lengths = pickle.load(f)
        else:
            for idx, (dir, subdir, _) in enumerate(
                    tqdm.tqdm(os.walk(folder), desc="Counting total number of frames")):
                if "view1" not in subdir:  # Only is 3 when the subdirs are the view subdirs
                    continue

                vid_path = os.path.join(dir, "view1")  # Do I want to use all views
                categ = dir.split('/')[-2]
                categ = class_dict[categ]

                if len(os.listdir(vid_path)) >= min_len:
                    self.videos.append((vid_path, categ))
                    self.lengths.append(len(os.listdir(vid_path)))

                # Hack because I'm lazy
                vid_path = os.path.join(dir, "view2")  # Do I want to use all views
                categ = dir.split('/')[-2]
                categ = class_dict[categ]

                if len(os.listdir(vid_path)) >= min_len:
                    self.videos.append((vid_path, categ))
                    self.lengths.append(len(os.listdir(vid_path)))

                vid_path = os.path.join(dir, "view3")  # Do I want to use all views
                categ = dir.split('/')[-2]
                categ = class_dict[categ]

                if len(os.listdir(vid_path)) >= min_len:
                    self.videos.append((vid_path, categ))
                    self.lengths.append(len(os.listdir(vid_path)))

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump((self.videos, self.lengths), f)

        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of frames {}".format(np.sum(self.lengths)))

    def __getitem__(self, item):
        return self.videos[item]

    def __len__(self):
        return len(self.videos)


# Will probably have to be converted to use NN
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, shuffle=True):
        self.dataset = dataset
        self.frames = []
        self.target = []
        for video, target in dataset:
            frames = sorted(os.listdir(video), key=lambda x: int(x.split('.')[0]))
            paths = [os.path.join(video, frame) for frame in frames]
            self.frames = self.frames + paths
            self.target = self.target + list(target * np.ones(len(os.listdir(video))))
        idx = np.random.permutation(len(self.frames))
        self.frames, self.target = np.array(self.frames), np.array(self.target)
        self.frames, self.target = self.frames[idx], self.target[idx]
        self.transforms = transform if transform is not None else lambda x: x

    def __getitem__(self, item):
        if item != 0:
            video_id = np.searchsorted(self.dataset.cumsum, item) - 1
            frame_num = item - self.dataset.cumsum[video_id] - 1
        else:
            video_id = 0
            frame_num = 0

        video, target = self.dataset[video_id]

        frames = sorted(os.listdir(video), key=lambda x: int(x.split('.')[0]))
        frame = np.array(Image.open(os.path.join(video, frames[frame_num])))
        '''
        pose_fname = glob.glob(f"{video}/../*.npy")[0]
        try:
            pose = np.load(pose_fname)[frame_num - 1]
        except IndexError:
            os.system(f"mv {video[:-5]} /scratch/PHPSDataset/broken_npy")
            pose = np.zeros((1, 24, 3))
            os.system("rm /scratch/PHPSDataset/HumanAct12/local.db")
        return {"images": self.transforms(frame), "categories": target, "pose": pose.flatten()}
        '''
        return {"image": self.transforms(frame), "categories": target}

    def __len__(self):
        return len(self.frames)


class_dict = {
    "A0101": 0,
    "A0102": 1,
    "A0103": 2,
    "A0104": 3,
    "A0105": 4,
    "A0106": 5,
    "A0107": 6,
    "A0201": 7,
    "A0301": 8,
    "A0401": 9,
    "A0402": 10,
    "A0501": 11,
    "A0502": 12,
    "A0503": 13,
    "A0504": 14,
    "A0505": 15,
    "A0601": 16,
    "A0602": 17,
    "A0603": 18,
    "A0604": 19,
    "A0605": 20,
    "A0701": 21,
    "A0801": 22,
    "A0802": 23,
    "A0803": 24,
    "A0901": 25,
    "A1001": 26,
    "A1002": 27,
    "A1101": 28,
    "A1102": 29,
    "A1103": 30,
    "A1104": 31,
    "A1201": 32,
    "A1202": 33,
}
