import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json
from PIL import Image
import torchvision.transforms.functional as T
from utils.rendering_utils import Camera, get_json_cameras

class BaseDataset(Dataset):

    def __init__(self, hair_folder:str, camera_folder:str, device:torch.device, data_type:str='synthetic', num_cameras:int=32, top_dir:bool=False, sample_index=[], flow_full=False):
        
        self.load_to_gpu = True  # Lazy Loading

        # Initially all tensors for all frames should lie in the strand directory in a folder called 'tensors' (Shape: (num_frames, 1, num_cameras, 1 , height, width))
        # If this is not set I will combine the information for all frames, If not a single frame is chosen
        self.top_dir = top_dir

        self.folder = hair_folder
        # Just a List of cases that have been rendered
        # Might be able to add the ability to just select one of the cases

        strands_list = os.listdir(hair_folder)

        self.hairstyles:list = [s for s in strands_list if s.startswith("strands")]

        self.flow_full = flow_full

        # the device where the data shall be stored
        self.device = device

        # We either have 'synthetic' Data as .dat files or pngs generated from 'real' captures
        self.data_type = data_type

        if data_type == 'synthetic':
            # Transformations fot the 3D Models
            self.model_tsfm_name = 'model_transform.data'
            self.bust_tsfm_name = 'bust_transform.data'

            # Map Names
            self.conf_name = 'conf.data' # Confidence for the Directional Map. Here just a Hair Mask with Black Bust
            self.dir_name = 'ori.data' # Directional Map
            self.depth_name = 'depth.data' # Hair and Bust Depth
            self.hair_depth_name = 'hair_depth.data' # With Black Bust
            self.bust_depth_name = 'bust_depth.data' # Without Hair
            self.flow_name = 'flow.data'
            self.flow_full_name = 'flow_full.data'
            self.mask_name = 'mask.data'
            self.hair_mask_name = 'mask_hair.data'
            self.bust_mask_name = 'mask_bust.data'

        else:
            raise RuntimeError('data type {} is not supported'.format(self.data_type))
        

        # Camera Stuff

        self.camera_folder = camera_folder # Basically the Folder where calibaration.json is

        self.camera_file = 'cameras.json' # Or caliration_dome.json

        self.num_views = num_cameras

        cams = get_json_cameras(os.path.join(self.camera_folder, self.camera_file))

        print('LEN: ', len(cams))

        # Extract camera extrinsics. Here as a extrinsic Matrix
        # cam.extrinsich is already a numpy array with shape [4,4]
        # TODO: Make sure this is consistent with the batshit definition that pytorch3d uses
        cam_poses = np.array([cam.extrinsic for cam in cams])
        self.cam_poses_c2w = torch.tensor(np.linalg.inv(cam_poses), dtype=torch.float32).reshape(-1, 4, 4).to(self.device)
        self.cam_poses_w2c = torch.tensor(cam_poses, dtype=torch.float32).reshape(-1, 4, 4).to(self.device)
        
        # Make sure all cameras have the same resolutions
        # TODO: Runtime error catching maybe
        resolutions = [[cam.width, cam.height] for cam in cams]
        
        imgsize = resolutions[0]

        for height, width in resolutions:
            if not [height, width] == imgsize:
                raise RuntimeError('All cameras need to have the same resolution!')
        
        self.img_size = imgsize # List with 2 entries [W,H]

        # Initialize "empty" ndc_proj for all cameras
        self.ndc_proj = torch.zeros((self.num_views, 4, 4), dtype=torch.float32)

        # Compute ndc_proj for each camera
        for i, cam_item in enumerate(cams):
            fx, fy = cam_item.fx, cam_item.fy
            cx, cy = cam_item.cx, cam_item.cy
            W, H = cam_item.width, cam_item.height

            # Compute NDC projection terms
            ndc_fx = fx / W
            ndc_fy = fy / H
            ndc_cx = (cx / W) - 0.5
            ndc_cy = (cy / H) - 0.5

            # Set near and far clipping planes (adjust as needed)
            n, f = 0.1, 100.0
            ndc_z0 = -(f + n) / (f - n)
            ndc_z1 = -2 * f * n / (f - n)

            # Construct ndc_proj for this camera
            self.ndc_proj[i, 0, 0] = ndc_fx
            self.ndc_proj[i, 1, 1] = ndc_fy
            self.ndc_proj[i, 0, 2] = ndc_cx
            self.ndc_proj[i, 1, 2] = ndc_cy
            self.ndc_proj[i, 2, 2] = ndc_z0
            self.ndc_proj[i, 2, 3] = ndc_z1
            self.ndc_proj[i, 3, 2] = -1.0

            self.ndc_proj = self.ndc_proj.to(self.device)

        #Batching


        if sample_index == []:
            # Build index mapping for DataLoader
            self.sample_index = []  # list of (hairstyle_id, start_frame)
            for strand_index, _ in enumerate(self.hairstyles):
                num_frames = self.get_num_frames(strand_index)
                for frame in range(num_frames):
                    self.sample_index.append((strand_index, frame))
        else:
            self.sample_index=sample_index

        # print(self.sample_index)

    def _read_synthetic_data(self, strand_index:int, frame_index:int): 
        '''
        read synthetic data, where imgs from different views are integrated for fast reading (also called union data)
        If frame is -1 all frames will be returned combined 
        Otherwise 0 is the first frame
        '''

        # index = [2,9,7,10,3,11,12,0,8,6,4,14,15,1,5,13] # WAS?
        case_path = os.path.join(self.folder, self.hairstyles[strand_index])

        frames = os.listdir(case_path)

        frames = [s for s in frames if s.startswith("frame")] #combined_tensors is in the same directory

        num_frames = len(frames) 

        frame_path = os.path.join(case_path, frames[frame_index])


        # # 1. model transform from open3D (TODO: Maybe need Transposition) and bust Transform
        # model_tsfm = torch.load(os.path.join(case_path,'combined_tensors' ,self.model_tsfm_name), weights_only=True)
        # model_tsfm = model_tsfm.reshape(4, 4)

        # bust_tsfm = torch.load(os.path.join(case_path,'combined_tensors' , self.bust_tsfm_name), weights_only=True).reshape(4, 4)

        # 2. Direction        
        # dir_union = torch.load(os.path.join(frame_path, "ori", self.dir_name), weights_only=True).reshape(1, self.num_views, 1, self.img_size[0], self.img_size[1])
        # dir_union = dir_union.half() / 255.0
        
        # 3. confidence        
        #     conf_union = torch.load(os.path.join(case_path,'combined_tensors', self.conf_name)).reshape(num_frames, self.num_views, 1, self.img_size[0], self.img_size[1])
        #     conf_union = torch.tensor(conf_union).clone().detach().half() / 255.0

        # 4. mask        
        # mask_union = torch.load(os.path.join(frame_path, 'mask', self.mask_name), weights_only=True).reshape(1 ,self.num_views, 1, self.img_size[0], self.img_size[1])
        # mask_union = mask_union.half() / 255.0
        # 5. depth
        # depth_union = torch.load(os.path.join(frame_path, 'depth', self.depth_name), weights_only=True).reshape(1 ,self.num_views, 1, self.img_size[0], self.img_size[1])
        # depth_union = depth_union.half() / 255.0

        # # 6. bust_depth
        # bust_depth_union = torch.load(os.path.join(case_path, 'combined_tensors', self.mask_name), weights_only=True).reshape(num_frames, self.num_views, 1, self.img_size[0], self.img_size[1]).half() / 255.0
            
        # 7. bust_mask
        # bust_mask_union = torch.load(os.path.join(frame_path, 'mask_bust', self.bust_mask_name), weights_only=True).reshape(1, self.num_views, 1, self.img_size[0], self.img_size[1]).half() / 255.0
        
        # 8. hair_mask
        hair_mask_union = torch.load(os.path.join(frame_path, 'mask_hair', self.hair_mask_name), weights_only=True).reshape(1, self.num_views, 1, self.img_size[0], self.img_size[1])
        hair_mask_union = hair_mask_union.half() / 255.0

        # 9. flow
        flow_union = torch.load(os.path.join(frame_path, 'flow', self.flow_name), weights_only=True).reshape(1, self.num_views, 1, self.img_size[0], self.img_size[1])
        flow_union = flow_union.half() / 255.0

        # 10. flow_full
        flow_full_union = torch.load(os.path.join(frame_path, 'flow_full', self.flow_full_name), weights_only=True).reshape(1, self.num_views, 2, self.img_size[0], self.img_size[1])
        flow_full_union = flow_full_union.half() / 255.0

        # depth_union = depth_union[index]
        item = {}
        item['hairstyle_id'] = self.hairstyles[strand_index] #.split('_')[1]
        # [4, 4]
        # item['model_tsfm'] = model_tsfm.to(self.device)
        # [4, 4]
        # item['bust_tsfm'] = bust_tsfm.to(self.device)
        # [F ,1 ,V, 1, H, W]
        # item['mask'] = mask_union.to(self.device)
        # [F ,1 ,V, 1, H, W]
        # item['hair_mask'] = hair_mask_union.to(self.device)
        # [F, 1, V, 2, H, W]
         #item['dir_map'] = (dir_union.to(self.device)) # * item['hair_mask'] #TODO: UNSURE ABOUT THIS
        # [F, 1, V, 1, H, W]
        # item['conf_map'] = conf_union.to(self.device)
        # [F, 1, V, 1, H, W]
        # item['depth_map'] = depth_union.to(self.device)
        # [F, 1, V, 1, H, W]
        # item['bust_depth_map'] = bust_depth_union.to(self.device)
        # [F, 1, V, 1, H, W]
        # item['hair_depth_map'] = hair_depth_union.to(self.device)
        # [F, 1, V, 1, H, W]
         #item['flow_map'] = flow_union.to(self.device)

        # item['mask'] = mask_union
        item['hair_mask'] = hair_mask_union
        # item['dir_map'] = dir_union
        # item['depth_map'] = depth_union
        item['flow_map'] = flow_union
        item['flow_full_map'] = flow_full_union #[F, 1, V, 2, H, W]


        # [V, 2, H, W]
        # item['orient_map'] = (orient_union.to(self.device) * 2. - 1.) * item['masks'] # Currently unused

        return item

    def get_num_frames(self, index) -> int:
        ''' Returns the amount of frames in a given index '''
        case_path = os.path.join(self.folder, self.hairstyles[index])

        frames_list = os.listdir(case_path)

        frames_list = [f for f in frames_list if f.startswith("frame")]

        return len(frames_list)
    
    def get_prev_idx_from_idx(self, idx) -> int:
        """Retruns either the prev frame idx or -1"""
        strand_index, frame_index = self.sample_index[idx]
        if frame_index == 0:
            return -1
        else:
            return idx-1

    def get_strand_idx_from_idx(self, idx)-> tuple:
        strand_index, frame_index = self.sample_index[idx]
        return (strand_index, frame_index)

    def __getitem__(self, idx):
        '''

        :param idx:
        :return:
        '''

        strand_index, frame_index = self.sample_index[idx]

        if self.data_type == 'synthetic':
            return self._read_synthetic_data(strand_index, frame_index=frame_index)
        # elif self.data_type == 'test':
        #     return self.read_test_data(idx)
        # elif self.data_type == 'real':
        #     return self.read_real_data(idx)
        else:
            raise RuntimeError('data type {} is not supported'.format(self.data_type))

    def __len__(self):
        return len(self.sample_index) # Returns the amount of hairstyles in the dataset


