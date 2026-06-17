import os
import numpy as np
import cv2
from torch.utils.data import Dataset

from utils_flow.img_processing_utils import pad_to_same_shape, pad_to_size, resize_keeping_aspect_ratio
from utils_flow.img_processing_utils import define_mask_zero_borders
from utils_flow.img_processing_utils import split2list


def valid_size(size):
    if not isinstance(size, (list)):
        size = [size, size]
    return size


class SingleImageDataset(Dataset):
    """MegaDepth dataset. Retrieves either pairs of matching images and their corresponding ground-truth flow
    (that is actually sparse) or single images. """
    def __init__(self, cfg, split='train', source_image_transform=None,
                 target_image_transform=None, flow_transform=None, co_transform=None, compute_mask_zero_borders=False,
                 store_scene_info_in_memory=False):
        """
        Args:
            root: root directory
            cfg: config (dictionary)
            split: 'train' or 'val'
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            co_transform: transforms to apply to both image pairs and corresponding flow field
            compute_mask_zero_borders: output mask of zero borders ?
            store_scene_info_in_memory: store all scene info in cpu memory? requires at least 50GB for training but
                                        sampling at each epoch is faster.

        Output in __getitem__:
        image
        """

        default_conf = {
            'seed': 400,
            'train_split': 'train_scenes_MegaDepth.txt',
            'train_debug_split': 'train_debug_scenes_MegaDepth.txt',
            'val_split': 'validation_scenes_MegaDepth.txt',
            'scene_info_path': '',
            'train_debug_num_per_scene': 10,
            'train_num_per_scene': 100,
            'val_num_per_scene': 25,

            'min_overlap_ratio': 0.3,
            'max_overlap_ratio': 1.,
            'max_scale_ratio': np.inf,

            'two_views': True,
            'exchange_images_with_proba': 0.5,
            'sort_by_overlap': False,
            
            'output_image_size': [520, 520],
            'pad_to_same_shape': True, 
            'output_flow_size': [[520, 520], [256, 256]],
            }
        
        self.cfg = default_conf
        self.cfg.update(cfg)

        self.image_data_path = self.cfg["image_data_path"]
        self.two_views = self.cfg['two_views']
        self.split = split

        self.output_image_size = valid_size(self.cfg['output_image_size'])

        # processing of final images
        self.source_image_transform = source_image_transform

        self.items = []
        # Make sure that the folders exist
        if not os.path.isdir(self.image_data_path):
            raise ValueError("the training directory path that you indicated does not exist ! ")
        for root, dirs, files in os.walk(self.image_data_path):
                for file in files:
                    self.items.append(os.path.join(root, file))
        

        print(' {} dataset comprises {} image pairs'.format(self.split, self.__len__()))


    def __len__(self):
        return len(self.items)


    def _read_single_view(self, path):
        image = cv2.imread(path)
        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = image[:, :, ::-1]  # go from BGR to RGB
        h, w, _ = image.shape
        if h > w:
            image = image.transpose((1, 0, 2))
        if self.output_image_size is not None:
            if isinstance(self.output_image_size, list):
                # resize to a fixed size and rescale the keypoints accordingly
                image = cv2.resize(image, (self.output_image_size[1], self.output_image_size[0]))

            else:
                # rescale both images so that the largest dimension is equal to the desired size of image and
                # then pad to obtain size 256x256 or whatever desired size. and change keypoints accordingly
                image, ratio_ = resize_keeping_aspect_ratio(image, self.output_image_size)
                image = pad_to_size(image, self.output_image_size)

        if self.source_image_transform is not None:
            image = self.source_image_transform(image)

        return {'image': image}

    def __getitem__(self, idx):
        """
        Args:
            idx

        Returns: Dictionary with fieldnames:
            image
        """
        # only retrieved a single image
        path = self.items[idx]
        output = self._read_single_view(path)
        return output
