import os
import torch
import numpy as np
from datasets.BaseDataset import BaseDataset

# Do not need utils
def getProjPoints(points, view, projection, tsfm=None):
    '''
    project 3D points on camera plane, return projected 2D locations (range [-1, 1])
    each view's contribution for each point, weighted by distance
    :param points: [4, N], N = num of points
    :param view: [V, 4, 4], V = num of views
    :param projection: [4, 4], projection matrix
    :param tsfm: [4, 4], model transformation
    :return: projected 2D points, [V, N, 1, 2]; each view's weight for each point, [V, 1, N]
    '''
    # [V, 4, N], world -> view -> projection
    # model_points = torch.matmul(tsfm, points) if tsfm is not None else points
    view_points = torch.matmul(view, points)
    proj_points = torch.matmul(projection, view_points)
    # divide coordinates by homogeneous coefficient to perform perspective projection
    xy_coord = (proj_points[:, :2, :] / proj_points[:, 3:4, :]).transpose(1, 2).unsqueeze(2)
    # flip y-axis because y positive in OpenGL screen space is opposite to y positive in tensor/img coordinate
    # xy_coord[..., 1] *= -1

    return xy_coord
    
def world_to_view(points_world, cam_poses_w2c):
    """
    points_world: [N, 4]
    cam_poses_w2c: [V, 4, 4]
    returns: [N, V, 3]
    """
    V = cam_poses_w2c.shape[0]
    points_cam = torch.matmul(cam_poses_w2c, points_world.T)  # [V,4,N]
    points_cam = points_cam[:, :3, :].permute(2, 0, 1)        # [N,V,3]
    return points_cam

class GT3DDataset(BaseDataset):

    def __init__(self, hair_folder:str, camera_folder:str, device, data_type='synthetic', num_cameras=32, resolution=256, num_pt_required=-1):
        super().__init__(hair_folder, camera_folder=camera_folder, device=device, data_type=data_type, num_cameras=num_cameras)

        # Everthing except +++resolution+++ and +++num_pt_required+++ is feeded to the superclass

        self.resolution = resolution

        self.camera_folder = camera_folder # Folder where camera.json is
        self.camera_fname = "cameras.json"

        self.voxel_points_name = 'voxel_points.data' # [N,6] First 3 are [X,Y,Z, ...] last 3 are direction can also be interpreted as color [...,R,G,B]. Insgesamt [X,Y,Z,R,G,B]
        # self.bust_tsfm = 'bust_transform.data' # Not used atm
        # self.hair_tsfm = 'hair_transform.data' # Not used atm

        self.num_pt_required = num_pt_required # If set, enforces a fixed number of points per sample (for batch consistency).
        # Mal sehen... ich habe so viele punkte...
        # Build index mapping for DataLoader
        

    def readOccSamplesFromFile(self, fname):
        data = torch.load(fname)
        
        coord = data[:, :3].T
        dirs = data[:, 3:6]
        flows = data[:, 6:]

        homo_coord = torch.tensor(np.concatenate([coord, np.ones((1, coord.shape[1]), dtype='float32')], axis=0))

        # print(homo_coord.shape)
        # print(homo_coord)


        return homo_coord, dirs, flows

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        strand_index, frame_index = self.sample_index[idx]

        if frame_index == 0:
            prev_frame_index = 0
        else:
            prev_frame_index = frame_index - 1

        case_path = os.path.join(self.folder, self.hairstyles[strand_index])

        frames = os.listdir(case_path)
        frames = [s for s in frames if s.startswith("frame")] #combined_tensors is in the same directory

        # current frame    
        frame_path = os.path.join(case_path, frames[frame_index])
        points_path = os.path.join(frame_path, 'mesh', self.voxel_points_name)
        points, dirs, flows = self.readOccSamplesFromFile(points_path) # [N,4] (homogenous) and [N,3], [N,3]

        # prev frame
        prev_frame_path = os.path.join(case_path, frames[prev_frame_index])
        prev_points_path = os.path.join(prev_frame_path, 'mesh', self.voxel_points_name)
        prev_points, prev_dirs, prev_flows = self.readOccSamplesFromFile(prev_points_path) # [N,4] (homogenous) and [N,3], [N,3]


        # randomly sample positive/negative points
        self.rand_perm = torch.randperm(dirs.shape[0]) #.to(self.device) 
        rand_perm = self.rand_perm

        # We need:

        # Global Coordinates
        # Prev global ccordinates

        # view coordinates
        # prev view coordinates 

        # view points
        # prev view points

        # GT connections for loss computation

        # [4, N]
        # Size is N*4*4 Bytes
        # With rougthly 50k points we have 1.2MB which is very acceptable and GPU (Nvidia GPU) friendly
        
        # --- projection to each view ---
        # assuming you have preloaded: self.views [V,4,4], self.projection [4,4]
        sample_coord = getProjPoints(
            points.float().to(self.device), 
            self.cam_poses_c2w, 
            self.ndc_proj
        )  #[V, N, 1, 2]

        prev_sample_coord = getProjPoints(
            prev_points.float().to(self.device), 
            self.cam_poses_c2w, 
            self.ndc_proj
        )  #[V, N_prev, 1, 2]

        # --- prepare tensors ---
        points = points.float()         # [4,N]
        prev_points = prev_points.float()  # [4,N_prev]

        pts_world = points.T[:, :3].float()
        prev_pts_world = prev_points.T[:, :3].float()
        gt_prev_pos = flows.float()         # [N,3]

        # from pytorch3d.ops import knn_points
        # # gt_prev_pos: [N, 3], prev_pts_world: [N_prev, 3]
        # # knn_points returns distances and indices for each point in gt_prev_pos to its k nearest neighbors in prev_pts_world
        # knn = knn_points(gt_prev_pos.unsqueeze(0), prev_pts_world.unsqueeze(0), K=1)
        # gt_conn_idx = knn.idx.squeeze(0).squeeze(-1)  # [N]

        pts_view = world_to_view(points.T.to(self.device), self.cam_poses_w2c)         # [N, V, 3]
        prev_pts_view = world_to_view(prev_points.T.to(self.device), self.cam_poses_w2c)  # [N_prev, V, 3]

        # Prev flow_full
        prev_flow_full_union = torch.load(os.path.join(prev_frame_path, 'flow_full', self.flow_full_name), weights_only=True).reshape(1, self.num_views, 2, self.img_size[0], self.img_size[1])
        prev_flow_full_union = prev_flow_full_union.half() / 255.0

        # Prev flow
        prev_flow_union = torch.load(os.path.join(prev_frame_path, 'flow', self.flow_name), weights_only=True).reshape(1, self.num_views, 1, self.img_size[0], self.img_size[1])
        prev_flow_union = prev_flow_union.half() / 255.0

        item['pts_view'] = pts_view
        item['prev_pts_view'] = prev_pts_view
        item['sample_coord'] = sample_coord
        item['pts_world'] = pts_world
        item['flow_view'] = prev_flows.unsqueeze(1).repeat(1, self.num_views, 1) # gt_prev_pos.unsqueeze(1).repeat(1, self.img_size[1], 1)  # optional, if you want view-wise copies 'TODO: Check img_size is correct
        # item['real_flow_view'] = gt_prev_pos.unsqueeze(1).repeat(1, self.num_views, 1)
        item['prev_pts_world'] = prev_pts_world
        item['prev_sample_coord'] = prev_sample_coord
        item['gt_prev_pos'] = gt_prev_pos     # [N,3]
        item['prev_flow_full_map'] = prev_flow_full_union
        item['prev_flow_map'] = prev_flow_union
        # item['gt_conn_idx'] = gt_conn_idx     # [N]

        real_flow_dir = pts_world - flows  # [N, 3]

        item['real_flow_view'] = real_flow_dir.unsqueeze(1).repeat(1, self.num_views, 1)

        return item


        # # Global Points
        # item['points'] = torch.tensor(points).to(self.device) # Randomly mixed points
        # item['prev_points'] = torch.tensor(prev_points).to(self.device) # Randomly mixed points

        # xy_coords = getProjPoints(item['points'], self.cam_poses_c2w, self.ndc_proj)

        # # [V, N, 1, 2]
        # item['xy_coords'] = xy_coords

        # item['sample_coords_view'] = xy_coords
        # if not self.num_pt_required == -1:
        #     item['sample_coords'] = item['sample_coords'][:self.num_pt_required]
        # # [N, 3]
        # # Directions
        # item['gt_dir_targets'] = dirs[rand_perm].long()
        # if not self.num_pt_required == -1:
        #     item['gt_dir_targets'] = item['gt_dir_targets'][:self.num_pt_required]

        # # [N,3]
        # item['gt_flow_targets'] = flows[rand_perm].long()
        # if not self.num_pt_required == -1:
        #     item['gt_flow_targets'] = item['gt_flow_targets'][:self.num_pt_required]

        # # [4, N]
        # # item['pts_world'] = item['model_tsfm'] @ item['points']
        # # [V, 4, N]
        # item['pts_view'] = self.cam_poses_c2w @ item['points']

        # # [N, 1, 3]
        # item['pts_world'] = item['points'][:3].transpose(0, 1).unsqueeze(1)
        # # [N, V, 3]
        # item['pts_view'] = item['pts_view'][:, :3, :].permute(2, 0, 1)

        # return item