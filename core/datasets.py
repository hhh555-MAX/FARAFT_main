# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import os
import math
import random
import h5py
import cv2
from tqdm import tqdm
from glob import glob
import os.path as osp
from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from utils.utils import induced_flow, check_cycle_consistency
from ddp_utils import *
from utils_flow.flow_and_mapping_operations import get_gt_correspondence_mask
from utils.utils import get_depth_tuple_transform_ops, get_tuple_transform_ops
import torchvision.transforms.functional as tvf
from utils.transforms import GeometricSequential
import kornia.augmentation as K
from utils.utils import warp_kpts

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        self.dataset = 'unknown'
        self.subsample_groundtruth = False
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.mask_list = []
        self.extra_info = []
        self.valid_list = []

    def __getitem__(self, index):
        while True:
            try:
                return self.fetch(index)
            except Exception as e:
                index = random.randint(0, len(self) - 1)

    def fetch(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        if self.sparse:
            if self.dataset == 'TartanAir':
                flow = np.load(self.flow_list[index])
                valid = np.load(self.mask_list[index])
                # rescale the valid mask to [0, 1]
                valid = 1 - valid / 100
                
            elif self.dataset == 'MegaDepth':
                depth0 = np.array(h5py.File(self.extra_info[index][0], 'r')['depth'])
                depth1 = np.array(h5py.File(self.extra_info[index][1], 'r')['depth'])
                camera_data = self.megascene[index]
                flow_01, flow_10 = induced_flow(depth0, depth1, camera_data)
                valid = check_cycle_consistency(flow_01, flow_10)
                flow = flow_01
            else:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            if self.dataset == 'Infinigen':
                # Inifinigen flow is stored as a 3D numpy array, [Flow, Depth]
                flow = np.load(self.flow_list[index])
                flow = flow[..., :2]
            elif self.dataset == 'dunhuang':
                flow = frame_utils.read_gen(self.flow_list[index])
                valid = get_gt_correspondence_mask(flow)
            elif self.dataset == 'sintel':
                flow = frame_utils.read_gen(self.flow_list[index])
                valid = frame_utils.read_gen(self.valid_list[index])
                valid = np.array(valid).astype(bool)
                valid = ~valid
            else:
                flow = frame_utils.read_gen(self.flow_list[index])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        
        if self.subsample_groundtruth:
            # use only every second value in both spatial directions ==> flow will have same dimensions as images
            # used for spring dataset
            flow = flow[::2, ::2]

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                if self.dataset == 'sintel' or self.dataset == 'dunhuang':
                    img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
                else:
                    img1, img2, flow = self.augmentor(img1, img2, flow)
        
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        flow[torch.isnan(flow)] = 0
        flow[flow.abs() > 1e9] = 0

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        valid_root = osp.join(root, split, 'occlusions')
        self.dataset = 'sintel'
        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))
                self.valid_list += sorted(glob(osp.join(valid_root, scene, '*.png')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1K'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1

class SpringFlowDataset(FlowDataset):
    """
    Dataset class for Spring optical flow dataset.
    For train, this dataset returns image1, image2, flow and a data tuple (framenum, scene name, left/right cam, FW/BW direction).
    For test, this dataset returns image1, image2 and a data tuple (framenum, scene name, left/right cam, FW/BW direction).

    root: root directory of the spring dataset (should contain test/train directories)
    split: train/test split
    subsample_groundtruth: If true, return ground truth such that it has the same dimensions as the images (1920x1080px); if false return full 4K resolution
    """
    def __init__(self, aug_params=None, root='datasets/spring', split='train', subsample_groundtruth=True):
        super(SpringFlowDataset, self).__init__(aug_params)

        assert split in ["train", "val", "test"]
        seq_root = os.path.join(root, split)

        if not os.path.exists(seq_root):
            raise ValueError(f"Spring {split} directory does not exist: {seq_root}")

        self.subsample_groundtruth = subsample_groundtruth
        self.split = split
        self.seq_root = seq_root
        self.data_list = []
        if split == 'test':
            self.is_test = True

        for scene in sorted(os.listdir(seq_root)):
            for cam in ["left", "right"]:
                images = sorted(glob(os.path.join(seq_root, scene, f"frame_{cam}", '*.png')))
                # forward
                for frame in range(1, len(images)):
                    self.data_list.append((frame, scene, cam, "FW"))
                # backward
                for frame in reversed(range(2, len(images)+1)):
                    self.data_list.append((frame, scene, cam, "BW"))

        for frame_data in self.data_list:
            frame, scene, cam, direction = frame_data

            img1_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame:04d}.png")

            if direction == "FW":
                img2_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame+1:04d}.png")
            else:
                img2_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame-1:04d}.png")

            self.image_list += [[img1_path, img2_path]]
            self.extra_info += [frame_data]

            if split != 'test':
                flow_path = os.path.join(self.seq_root, scene, f"flow_{direction}_{cam}", f"flow_{direction}_{cam}_{frame:04d}.flo5")
                self.flow_list += [flow_path]

class Infinigen(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/infinigen'):
        super(Infinigen, self).__init__(aug_params)
        self.root = root
        scenes = glob(osp.join(self.root, '*/*/'))
        self.dataset = "Infinigen"
        for scene in sorted(scenes):
            if not osp.isdir(osp.join(scene, 'frames')):
                continue
            images = sorted(glob(osp.join(scene, 'frames/Image/camera_0/*.png')))
            for idx in range(len(images) - 1):
                # name = Image + "_{ID}"
                ID = images[idx].split('/')[-1][6:-4]
                self.image_list.append([images[idx], images[idx + 1]])
                self.flow_list.append(osp.join(scene, 'frames/Flow3D/camera_0', f"Flow3D_{ID}.npy"))

class TartanAir(FlowDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, aug_params=None, root='datasets/tartanair'):
        super(TartanAir, self).__init__(aug_params, sparse=True)
        self.n_frames = 2
        self.dataset = 'TartanAir'
        self.root = root
        self._build_dataset()

    def _build_dataset(self):
        scenes = glob(osp.join(self.root, '*/*/*/*/*/*'))
        for scene in sorted(scenes):
            images = sorted(glob(osp.join(scene, 'image_left/*.png')))
            for idx in range(len(images) - 1):
                frame0 = str(idx).zfill(6)
                frame1 = str(idx + 1).zfill(6)
                self.image_list.append([images[idx], images[idx + 1]])
                self.flow_list.append(osp.join(scene, 'flow', f"{frame0}_{frame1}_flow.npy"))
                self.mask_list.append(osp.join(scene, 'flow', f"{frame0}_{frame1}_mask.npy"))



class MegadepthScene:
    def __init__(
        self,
        data_root,
        scene_info,
        ht=384,
        wt=512,
        min_overlap=0.0,
        shake_t=0,
        rot_prob=0.0,
        normalize=False,
        return_data_dict = False
    ) -> None:
        self.data_root = data_root
        self.image_paths = scene_info["image_paths"]
        self.depth_paths = scene_info["depth_paths"]
        self.intrinsics = scene_info["intrinsics"]
        self.poses = scene_info["poses"]
        self.pairs = scene_info["pairs"]
        self.overlaps = scene_info["overlaps"]
        threshold = self.overlaps > min_overlap
        self.pairs = self.pairs[threshold]
        self.overlaps = self.overlaps[threshold]
        self.return_data_dict = return_data_dict
        if len(self.pairs) > 100000:
            pairinds = np.random.choice(
                np.arange(0, len(self.pairs)), 100000, replace=False
            )
            self.pairs = self.pairs[pairinds]
            self.overlaps = self.overlaps[pairinds]
        # counts, bins = np.histogram(self.overlaps,20)
        # print(counts)
        self.im_transform_ops = get_tuple_transform_ops(
            resize=(ht, wt), normalize=normalize, unscale=True
        )
        self.depth_transform_ops = get_depth_tuple_transform_ops(
            resize=(ht, wt), normalize=False
        )
        self.wt, self.ht = wt, ht
        self.shake_t = shake_t
        self.H_generator = GeometricSequential(K.RandomAffine(degrees=90, p=rot_prob))

    def load_im(self, im_ref, crop=None):
        im = Image.open(im_ref)
        return im

    def load_depth(self, depth_ref, crop=None):
        depth = np.array(h5py.File(depth_ref, "r")["depth"])
        return torch.from_numpy(depth)

    def __len__(self):
        return len(self.pairs)

    def scale_intrinsic(self, K, wi, hi):
        sx, sy = self.wt / wi, self.ht / hi
        sK = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        return sK @ K

    def rand_shake(self, *things):
        t = np.random.choice(range(-self.shake_t, self.shake_t + 1), size=2)
        return [
            tvf.affine(thing, angle=0.0, translate=list(t), scale=1.0, shear=[0.0, 0.0])
            for thing in things
        ], t

    def flow_and_certainty(self, depth1, depth2, T_1to2, K1, K2, query_img):
        """[summary]

        Args:
            H ([type]): [description]
            scale ([type]): [description]

        Returns:
            [type]: [description]
        """
        b, d, h1, w1 = query_img.shape
        with torch.no_grad():
            x1_n = torch.meshgrid(
                *[
                    torch.linspace(
                        -1 + 1 / n, 1 - 1 / n, n, device=query_img.device
                    )
                    for n in (b, h1, w1)
                ]
            )
            x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(b, h1 * w1, 2)
            mask, x2 = warp_kpts(
                x1_n.double(),
                depth1.double(),
                depth2.double(),
                T_1to2.double(),
                K1.double(),
                K2.double(),
            )
            prob = mask.float().reshape(b, h1, w1)
            x2 = x2.reshape(b, h1, w1, 2)
            x2 = torch.clamp(x2, max=1, min=-1)
            x, y = np.meshgrid(np.arange(w1), np.arange(h1), indexing='xy')
            coords0 = np.stack([x, y], axis=-1)[None]
            coords1 = torch.stack((w1/2 * (x2[...,0]+1), h1/2 * (x2[...,1]+1)),axis=-1)
            flow = coords1 - coords0
            flow = flow.permute(0, 3, 1, 2)
            flow[torch.isnan(flow)] = 0
            flow[flow.abs() > 1e9] = 0
        return flow, prob

    def __getitem__(self, index):
        while True:
            try:
                return self.fetch(index)
            except Exception as e:
                index = random.randint(0, len(self) - 1)

    def fetch(self, pair_idx):
        # read intrinsics of original size
        idx1, idx2 = self.pairs[pair_idx]
        K1 = torch.tensor(self.intrinsics[idx1].copy(), dtype=torch.float).reshape(3, 3)
        K2 = torch.tensor(self.intrinsics[idx2].copy(), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T1 = self.poses[idx1]
        T2 = self.poses[idx2]
        T_1to2 = torch.tensor(np.matmul(T2, np.linalg.inv(T1)), dtype=torch.float)[
            :4, :4
        ]  # (4, 4)

        # Load positive pair data
        im1, im2 = self.image_paths[idx1], self.image_paths[idx2]
        depth1, depth2 = self.depth_paths[idx1], self.depth_paths[idx2]
        im_src_ref = os.path.join(self.data_root, im1)
        im_pos_ref = os.path.join(self.data_root, im2)
        depth_src_ref = os.path.join(self.data_root, depth1)
        depth_pos_ref = os.path.join(self.data_root, depth2)
        # return torch.randn((1000,1000))
        im_src = self.load_im(im_src_ref)
        im_pos = self.load_im(im_pos_ref)
        depth_src = self.load_depth(depth_src_ref)
        depth_pos = self.load_depth(depth_pos_ref)

        # Recompute camera intrinsic matrix due to the resize
        K1 = self.scale_intrinsic(K1, im_src.width, im_src.height)
        K2 = self.scale_intrinsic(K2, im_pos.width, im_pos.height)
        # Process images
        im_src, im_pos = self.im_transform_ops((im_src, im_pos))
        depth_src, depth_pos = self.depth_transform_ops(
            (depth_src[None, None], depth_pos[None, None])
        )
        [im_src, im_pos, depth_src, depth_pos], t = self.rand_shake(
            im_src, im_pos, depth_src, depth_pos
        )
        im_src, Hq = self.H_generator(im_src[None])
        depth_src = self.H_generator.apply_transform(depth_src, Hq)
        K1[:2, 2] += t
        K2[:2, 2] += t
        K1 = Hq[0] @ K1
        data_dict = {
            "query": im_src[0],
            "query_identifier": self.image_paths[idx1].split("/")[-1].split(".jpg")[0],
            "support": im_pos,
            "support_identifier": self.image_paths[idx2]
            .split("/")[-1]
            .split(".jpg")[0],
            "query_depth": depth_src[0, 0],
            "support_depth": depth_pos[0, 0],
            "K1": K1,
            "K2": K2,
            "T_1to2": T_1to2}

        if self.return_data_dict == True:
            return data_dict
        else:
            flow, prob = self.flow_and_certainty(
                data_dict["query_depth"][None],
                data_dict["support_depth"][None],
                data_dict["T_1to2"][None],
                data_dict["K1"][None],
                data_dict["K2"][None],
                data_dict['query'][None])
            return im_src[0], im_pos, flow[0], prob[0]


class MegadepthBuilder:
    def __init__(self, data_root="datasets/megadepth") -> None:
        self.data_root = data_root
        self.scene_info_root = os.path.join(data_root, "prep_scene_info_no_sfm")
        self.all_scenes = os.listdir(self.scene_info_root)
        self.test_scenes = ["0017.npy", "0004.npy", "0048.npy", "0013.npy"]
        self.test_scenes_loftr = ["0015.npy", "0022.npy"]


    def build_scenes(self, split="train", min_overlap=0.0, **kwargs):
        if split == "train":
            scene_names = set(self.all_scenes) - set(self.test_scenes)
        elif split == "train_loftr":
            scene_names = set(self.all_scenes) - set(self.test_scenes_loftr)
        elif split == "test":
            scene_names = self.test_scenes
        elif split == "test_loftr":
            scene_names = self.test_scenes_loftr
        else:
            raise ValueError(f"Split {split} not available")
        scenes = []
        for scene_name in scene_names:
            scene_info = np.load(
                os.path.join(self.scene_info_root, scene_name), allow_pickle=True
            ).item()
            scenes.append(
                MegadepthScene(
                    self.data_root, scene_info, min_overlap=min_overlap, **kwargs
                )
            )
        return scenes

    def weight_scenes(self, concat_dataset, alpha=0.5):
        ns = []
        for d in concat_dataset.datasets:
            ns.append(len(d))
        ws = torch.cat([torch.ones(n) / n**alpha for n in ns])
        return ws


class Middlebury(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/middlebury'):
        super(Middlebury, self).__init__(aug_params)
        img_root = os.path.join(root, 'images')
        flow_root = os.path.join(root, 'flow')

        flows = []
        imgs = []
        info = []

        for scene in sorted(os.listdir(flow_root)):
            img0 = os.path.join(img_root, scene, "frame10.png")
            img1 = os.path.join(img_root, scene, "frame11.png")
            flow = os.path.join(flow_root, scene, "flow10.flo") 
            imgs += [(img0, img1)]
            flows += [flow]
            info += [scene]

        self.image_list = imgs
        self.flow_list = flows
        self.extra_info = info

# TODO 制作数据集时需要改动这里
class dunhuang(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/dunhuang', dstype='train'):
        super(dunhuang, self).__init__(aug_params)
        if split == 'testing':
            self.is_test = True
        self.dataset = 'dunhuang'
        images1 = sorted(glob(osp.join(root, dstype, 'images/*_img_2.jpg')))
        images2 = sorted(glob(osp.join(root, dstype, 'images/*_img_1.jpg')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, dstype, 'flow/*.flo')))

def fetch_dataloader(args, rank=0, world_size=1, use_ddp=False):
    """ Create the data loader for the corresponding trainign set """

    if args.dataset == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.1, 'max_scale': args.scale + 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.dataset == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.4, 'max_scale': args.scale + 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.dataset == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.2, 'max_scale': args.scale + 0.6, 'do_flip': True}
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        train_dataset = sintel_clean + sintel_final

    elif args.dataset == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.2, 'max_scale': args.scale + 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    elif args.dataset == 'dunhuang':
        if args.train_dstype == 'mix':
            homo = dunhuang(split='training', dstype='train')
            tps = dunhuang(split='training', dstype='train+tps')
            train_dataset = homo + tps
        else:
            train_dataset = dunhuang(split='training', dstype=args.train_dstype)

    elif args.dataset == 'dunhuang_raft':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        train_dataset = 10 * dunhuang(aug_params, split='training', dstype=args.train_dstype)

    elif args.dataset == 'spring':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale, 'max_scale': args.scale + 0.2, 'do_flip': True}
        train_dataset = SpringFlowDataset(aug_params, subsample_groundtruth=True)

    elif args.dataset == 'TartanAir':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.2, 'max_scale': args.scale + 0.4, 'do_flip': True}
        train_dataset = TartanAir(aug_params)
    
    elif args.dataset == 'TSKH':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.2, 'max_scale': args.scale + 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        kitti = KITTI({'crop_size': args.image_size, 'min_scale': args.scale - 0.3, 'max_scale': args.scale + 0.5, 'do_flip': True})
        hd1k = HD1K({'crop_size': args.image_size, 'min_scale': args.scale - 0.5, 'max_scale': args.scale + 0.2, 'do_flip': True})
        train_dataset = 20 * sintel_clean + 20 * sintel_final + 80 * kitti + 30 * hd1k + things
    
    elif args.dataset == 'TSKH_raft':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        
        kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
        hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
        train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things #与上面区别在于比例不同

    elif args.dataset == 'SD':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        
        Dh = dunhuang(aug_params, split='training', dstype=args.train_dstype)
        train_dataset = 10*Dh + 5*sintel_clean + 5*sintel_final

    elif args.dataset == 'megadepth':
        # train_dataset = MegaDepth(dstype='train')
        mega = MegadepthBuilder(data_root="datasets/megadepth")
        
        # megadepth_train1 = mega.build_scenes(
        #     split="train_loftr", min_overlap=0.01, ht=args.image_size[0], wt=args.image_size[1], shake_t=32
        # )
        megadepth_train2 = mega.build_scenes(
            split="train_loftr", min_overlap=0.5, ht=args.image_size[0], wt=args.image_size[1], shake_t=32
        )
        # train_dataset = ConcatDataset(megadepth_train1 + megadepth_train2)
        train_dataset = ConcatDataset(megadepth_train2)
        # mega_ws = mega.weight_scenes(megadepth_train, alpha=0.75)

    elif args.dataset == 'TKH':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.4, 'max_scale': args.scale + 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        kitti = KITTI({'crop_size': args.image_size, 'min_scale': args.scale - 0.3, 'max_scale': args.scale + 0.5, 'do_flip': True})
        hd1k = HD1K({'crop_size': args.image_size, 'min_scale': args.scale - 0.5, 'max_scale': args.scale + 0.2, 'do_flip': True})
        train_dataset = 100 * hd1k + clean_dataset + final_dataset + 1000 * kitti

    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank)
        num_gpu = torch.cuda.device_count()
        print('dataloader_num_gpu:'+str(num_gpu))
        # train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size // num_gpu, 
        #     shuffle=(train_sampler is None), num_workers=calc_num_workers(), sampler=train_sampler, worker_init_fn=init_fn)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size // num_gpu, 
            shuffle=(train_sampler is None), num_workers=4, sampler=train_sampler, worker_init_fn=init_fn, drop_last=True)
    else:
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
            pin_memory=False, shuffle=True, num_workers=4, drop_last=True) #这个地方的workers原来是32

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

