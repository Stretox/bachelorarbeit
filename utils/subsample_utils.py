import torch
import math
import numpy as np

# Methods for generating subsamples

def subsample(batch, device, num_points=1000, num_prev_points=1000, debug=False):
    """
    Input: 
        batch is the dictionary from the DataLoader
        device is cuda or cpu
        num_points are the amount of query points or 0 to get all
        num_prev_points are the num closest points to each num point or 0 for all
        debug: print the shapes
    Returns: 
        a dict with subsampled points on device
    """

    # Make local copies
    # If not "get" then required
    sample_coord = batch['sample_coord'] 
    prev_sample_coord = batch['prev_sample_coord']
    pts_world = batch['pts_world'] # [N,3]
    prev_pts_world = batch['prev_pts_world']
    pts_view = batch.get('pts_view', None)
    prev_pts_view = batch.get('prev_pts_view', None)
    prev_flow_view = batch['flow_view']
    real_flow_view = batch['real_flow_view']
    gt_prev_pos = batch['gt_prev_pos']
    gt_conn_idx = batch.get('gt_conn_idx', None)
    
    if debug:
        # Print the shapes of all variables
        print("sample_coord:", sample_coord.shape)
        print("prev_sample_coord:", prev_sample_coord.shape)
        print("pts_world:", pts_world.shape)
        print("prev_pts_world:", prev_pts_world.shape)
        print("pts_view:", pts_view.shape if pts_view is not None else "None")
        print("prev_pts_view:", prev_pts_view.shape if prev_pts_view is not None else "None")
        print("prev_flow_view:", prev_flow_view.shape)
        print("real_flow_view:", real_flow_view.shape)
        print("gt_prev_pos:", gt_prev_pos.shape)
        print("gt_conn_idx:", gt_conn_idx.shape if gt_conn_idx is not None else "None")

    ################ Shape fixing #################

    # Fix shapes of sample coordinates
    sample_coord = sample_coord.squeeze(0)
    prev_sample_coord = prev_sample_coord.squeeze(0)

    pts_world = pts_world.squeeze(0)
    prev_pts_world = prev_pts_world.squeeze(0)

    gt_prev_pos = gt_prev_pos.squeeze(0)

    pts_view = pts_view.squeeze(0)
    prev_pts_view = prev_pts_view.squeeze(0)

    prev_flow_view = prev_flow_view.squeeze(0)
    real_flow_view = real_flow_view.squeeze(0)

    ################ Sampling #####################

    N = pts_world.shape[0]
    M = prev_pts_world.shape[0]

    if num_points <= 0 or N <= num_points:
        idx = torch.arange(N, device=device)
    else:
        idx = torch.randperm(N, device=device)[:num_points]

    if num_prev_points <= 0 or M <= num_prev_points:
        prev_idx = torch.arange(M, device=device)
    else:
        # get only the num_prev_points closest points for every query point
        sub_pts_world = pts_world[idx]  # [num_points, 3]
        dists = torch.cdist(sub_pts_world, prev_pts_world, p=2)  # [num_points, M]
        _, knn_idx = torch.topk(dists, k=num_prev_points, dim=1, largest=False)  # [num_points, k]
        prev_idx = knn_idx.reshape(-1)

    # ensure sample-coords are on device
    sample_coord = sample_coord.to(device)
    prev_sample_coord = prev_sample_coord.to(device)

    # sample points from both
    sample_coord = sample_coord[:, idx, :, :].contiguous()

    prev_sample_coord = prev_sample_coord[:, prev_idx, :, :].contiguous()

    # pts_world/prev_pts_world on device and subsampled
    pts_world = pts_world.to(device)[idx].contiguous()
    prev_pts_world = prev_pts_world.to(device)[prev_idx].contiguous()

    # pts_view/prev_pts_view: assume shape [N, V, 3] and [N_prev, V, 3]
    pts_view = pts_view.to(device)[idx].contiguous()
    prev_pts_view = prev_pts_view.to(device)[prev_idx].contiguous()

    prev_flow_view = prev_flow_view[prev_idx.to(device='cpu')].to(device).contiguous()

    # gt_prev_pos and gt_conn_idx
    gt_prev_pos = gt_prev_pos.to(device)[idx].contiguous()
    if gt_conn_idx is not None:
        gt_conn_idx = gt_conn_idx.to(device)[idx].contiguous()

    # return a cleaned dict
    return dict(
        sample_coord=sample_coord,           # [V, n_sub, 1, 2]
        pts_world=pts_world,                 # [n_sub, 3]
        pts_view=pts_view,                   # [n_sub, V, 3]
        prev_flow_view=prev_flow_view,       # [n_sub, V, 3]
        real_flow_view = real_flow_view,
        prev_sample_coord=prev_sample_coord, # [V, m_sub, 1, 2]
        prev_pts_world=prev_pts_world,       # [m_sub, 3]
        prev_pts_view=prev_pts_view,         # [m_sub, V, 3]
        gt_prev_pos=gt_prev_pos,             # [n_sub, 3]
        gt_conn_idx=gt_conn_idx
    )

def subsample_iter(batch, device, num_points:int=1000, num_prev_points:int=1000, debug=False):
    """
    Yield subsampled chunks covering all points in a batch.
    num_points is the number of query points for training.
    num_prev_points is the number of prev_points closest to num_points for training.
    """
    # First get the full samples
    full_sample = subsample(batch, device, num_points=0, num_prev_points=0, debug=debug)

    # extract full tensors
    sample_coord_full = full_sample['sample_coord']       # [V, N, 1, 2]
    prev_sample_coord_full = full_sample['prev_sample_coord']  # [V, M, 1, 2]
    pts_world_full = full_sample['pts_world']             # [N, 3]
    prev_pts_world_full = full_sample['prev_pts_world']   # [M, 3]
    pts_view_full = full_sample['pts_view'] 
    prev_pts_view_full = full_sample['prev_pts_view']  
    flow_view_full = full_sample['prev_flow_view'] 
    real_flow_view_full = full_sample['real_flow_view']
    gt_prev_pos_full = full_sample['gt_prev_pos']# [N,3]

    # Ensure everything is on the requested device
    def to_device(t):
        return t.to(device) if t is not None else None

    sample_coord_full = to_device(sample_coord_full)
    prev_sample_coord_full = to_device(prev_sample_coord_full)
    pts_world_full = to_device(pts_world_full)
    prev_pts_world_full = to_device(prev_pts_world_full)
    pts_view_full = to_device(pts_view_full)
    prev_pts_view_full = to_device(prev_pts_view_full)
    flow_view_full = to_device(flow_view_full)
    real_flow_view_full = to_device(real_flow_view_full)
    gt_prev_pos_full = to_device(gt_prev_pos_full)

    N = pts_world_full.shape[0]
    M = prev_pts_world_full.shape[0]

    # if num_points == 0 yield once with everything
    if num_points <= 0 or N <= num_points:
        yield dict(
            sample_coord=sample_coord_full.contiguous(),
            pts_world=pts_world_full.contiguous(),
            pts_view=pts_view_full.contiguous(),
            flow_view=flow_view_full.contiguous(),
            real_flow_view=real_flow_view_full.contiguous(),
            prev_sample_coord=prev_sample_coord_full.contiguous(),
            prev_pts_world=prev_pts_world_full.contiguous(),
            prev_pts_view=prev_pts_view_full.contiguous(),
            gt_prev_pos=gt_prev_pos_full.contiguous(),
        )
        return

    # compute number of chunks (use ceil so we cover remainder)
    num_chunks = int(math.ceil(N / float(num_points)))
    prev_num_chunks = int(math.ceil(M / float(num_prev_points))) if num_prev_points > 0 else 1 # Unused

    # helper to safe-index along first dim
    def safe_index(tensor, idx_tensor):
        """
        Returns index tensor along its first dimension with idx_tensor.
        Moves idx_tensor to tensor.device before indexing.
        """
        if tensor is None:
            return None
        idx_tensor = idx_tensor.to(tensor.device)
        return tensor[idx_tensor].contiguous()

    for i in range(num_chunks):
        start = i * num_points
        end = min((i + 1) * num_points, N)
        idx = torch.arange(start, end, dtype=torch.long, device=device)

        sub_pts_world = pts_world_full[idx]  # [num_points, 3]
        dists = torch.cdist(sub_pts_world, prev_pts_world_full, p=2)  # [num_points, M]
        _, knn_idx = torch.topk(dists, k=num_prev_points, dim=1, largest=False)  # [num_points, k]
        prev_idx = knn_idx.reshape(-1)

        if prev_idx.numel() < num_points * num_prev_points:
            pad_needed = num_points * num_prev_points - prev_idx.numel()
            pad = prev_idx[:pad_needed % prev_idx.numel()]
            prev_idx = torch.cat([prev_idx, pad], dim=0)

        # sample current and prev
        sample_coord = sample_coord_full[:, idx, :, :].contiguous()
        pts_world = pts_world_full[idx].contiguous()
        pts_view = safe_index(pts_view_full, idx)
        gt_prev_pos = safe_index(gt_prev_pos_full, idx)
        
        # pts_world and related are part of current points
        pts_world = safe_index(pts_world_full, idx)
        pts_view = safe_index(pts_view_full, idx)
        gt_prev_pos = safe_index(gt_prev_pos_full, idx)
        real_flow_view = safe_index(real_flow_view_full, idx)

        flow_view = safe_index(flow_view_full, prev_idx)

        # prev-frame items
        prev_sample_coord = prev_sample_coord_full[:, prev_idx, :, :].contiguous() 
        prev_pts_world = safe_index(prev_pts_world_full, prev_idx)
        prev_pts_view = safe_index(prev_pts_view_full, prev_idx)

        yield dict(
            sample_coord=sample_coord,
            pts_world=pts_world,
            pts_view=pts_view,
            flow_view=flow_view,
            real_flow_view=real_flow_view,
            prev_sample_coord=prev_sample_coord,
            prev_pts_world=prev_pts_world,
            prev_pts_view=prev_pts_view,
            gt_prev_pos=gt_prev_pos,
        )

