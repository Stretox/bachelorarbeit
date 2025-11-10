import torch
import numpy as np


def canonicalize_and_subsample(batch, device, num_points=1000, num_prev_points=1000):
    """
    Input: batch is the dictionary from DataLoader (may have extra batch dims).
    Returns: a dict with canonical shapes and subsampled points on `device`.
    """

    def to_device(t):
        return t.to(device) if isinstance(t, torch.Tensor) else t

    # Make local copies
    sample_coord = batch['sample_coord']          # may be [V,N,1,2] OR [1,V,N,1,2] OR [1,32,N,1,2]
    prev_sample_coord = batch['prev_sample_coord']
    pts_world = batch['pts_world']                # expected [N,3]
    prev_pts_world = batch['prev_pts_world']
    pts_view = batch.get('pts_view', None)
    prev_pts_view = batch.get('prev_pts_view', None)
    flow_view = batch['flow_view']
    real_flow_view = batch['real_flow_view']
    gt_prev_pos = batch['gt_prev_pos']
    gt_conn_idx = batch.get('gt_conn_idx', None)

    # Print the shapes of all variables
    # print("sample_coord:", sample_coord.shape)
    # print("prev_sample_coord:", prev_sample_coord.shape)
    # print("pts_world:", pts_world.shape)
    # print("prev_pts_world:", prev_pts_world.shape)
    # print("pts_view:", pts_view.shape if pts_view is not None else "None")
    # print("prev_pts_view:", prev_pts_view.shape if prev_pts_view is not None else "None")
    # print("flow_view:", flow_view.shape)
    # print("gt_prev_pos:", gt_prev_pos.shape)
    # print("gt_conn_idx:", gt_conn_idx.shape if gt_conn_idx is not None else "None")

    # Accept typical variants: [V,N,1,2], [1,V,N,1,2], [V,1,N,1,2], [1, V, N, 1, 2], [V,N,1,2,1] (rare)
    def canonicalize_sample_coord(sc):
        if isinstance(sc, np.ndarray):
            sc = torch.from_numpy(sc)
        if sc.dim() == 5 and sc.shape[0] == 1:
            sc = sc.squeeze(0)       # [V, N, 1, 2] or [V, N, 1, 2]
        # if shape is [1, V, N, 1,2] do squeeze 0
        if sc.dim() == 5 and sc.shape[1] != 2 and sc.shape[0] != sc.shape[1]:
            # handle [1, V, N, 1, 2]
            if sc.shape[0] == 1:
                sc = sc.squeeze(0)
        # final check
        if sc.dim() != 4:
            raise RuntimeError(f"sample_coord has unexpected dim {sc.dim()}, shape {tuple(sc.shape)}")
        # ensure order [V, N, 1, 2]
        return sc

    sample_coord = canonicalize_sample_coord(sample_coord)
    prev_sample_coord = canonicalize_sample_coord(prev_sample_coord)

    def canonicalize_pts_world(p):
        if isinstance(p, np.ndarray):
            p = torch.from_numpy(p)
        # strip leading batch dim 1 if present
        if p.dim() == 3 and p.shape[0] == 1:
            # e.g., [1, 4, 3] or [1, N, 3]
            p = p.squeeze(0)
        # sometimes pts are [4, N] (homogeneous) or [4,N,1]
        if p.dim() == 2 and p.shape[0] == 4:
            # convert [4,N] -> [N,4] then drop last column if homogeneous
            p = p.T
        if p.dim() == 2 and p.shape[1] == 4:
            # [N,4] keep only xyz
            p = p[:, :3]
        if p.dim() == 2 and p.shape[1] == 3:
            return p
        if p.dim() == 1:
            # single point -> [1,3] not expected but handle
            return p.unsqueeze(0)
        if p.dim() == 3:
            # maybe [N, V, 3] passed accidentally
            # try to reduce to [N,3] by selecting first view or raise
            if p.shape[1] in (3,4):  # suspicious ordering
                p = p.reshape(-1, p.shape[-1])
                return p
        raise RuntimeError(f"Unexpected pts_world shape: {tuple(p.shape)}")

    pts_world = canonicalize_pts_world(pts_world)
    prev_pts_world = canonicalize_pts_world(prev_pts_world)
    gt_prev_pos = torch.from_numpy(gt_prev_pos) if isinstance(gt_prev_pos, np.ndarray) else gt_prev_pos
    gt_prev_pos = gt_prev_pos.squeeze(0) if gt_prev_pos.dim() == 3 and gt_prev_pos.shape[0] == 1 else gt_prev_pos
    if gt_prev_pos.dim() == 2 and gt_prev_pos.shape[1] == 3:
        pass
    else:
        # try different squeezes- SQUEEZE SQUEEZE
        gt_prev_pos = gt_prev_pos.reshape(-1, 3)

    def canonicalize_pts_view(pv, expected_N=None):
        if pv is None:
            return None
        if isinstance(pv, np.ndarray):
            pv = torch.from_numpy(pv)
        # possible shapes seen: [1, N, V, 3] or [1, N, 32, 3] or [N, V, 3]
        if pv.dim() == 4 and pv.shape[0] == 1:
            pv = pv.squeeze(0)  # [N, V, 3]
        # if it's [N, V, 3] we're good
        if pv.dim() == 3:
            # sometimes it's [1, N, V] etc; try to permute if second dim equals num_views
            return pv
        # if pv is [N, 3, V] or [N, V, 3] etc. try to reshape
        pv = pv.reshape(-1, pv.shape[-2], pv.shape[-1])
        return pv

    pts_view = canonicalize_pts_view(pts_view)
    prev_pts_view = canonicalize_pts_view(prev_pts_view)

    # if isinstance(flow_view, np.ndarray):
    #     flow_view = torch.from_numpy(flow_view)
    # # handle [1, N, V, 3] or [N, V, 3] or [1, N, 4608, 3] (bad) -> we must ensure V matches num views
    # if flow_view.dim() == 4 and flow_view.shape[0] == 1:
    #     flow_view = flow_view.squeeze(0)
    # # if flow_view dims don't match pts_world/pts_view, we will raise later.

    N = pts_world.shape[0]
    M = prev_pts_world.shape[0]

    if num_points <= 0 or N <= num_points:
        idx = torch.arange(N, device=device)
    else:
        idx = torch.randperm(N, device=device)[:num_points]

    if num_prev_points <= 0 or M <= num_prev_points:
        prev_idx = torch.arange(M, device=device)
    else:
        prev_idx = torch.randperm(M, device=device)[:num_prev_points]

    def subset_first_dim(tensor, idx):
        if tensor is None:
            return None
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        # handle sample_coord separately (its axis ordering is [V,N,1,2] or [1,V,N,1,2])
        return tensor[idx] if tensor.dim() >= 1 and tensor.shape[0] == N else tensor

    # sample_coord is [V, N, 1, 2] or maybe [1, V, N, 1,2]; we canonicalized earlier
    # ensure it ins on device
    sample_coord = sample_coord.to(device)
    prev_sample_coord = prev_sample_coord.to(device)

    # sample along N axis (axis 1 if sample_coord is [V, N, 1, 2], or axis 2 if it's [V, 1, N, 2] earlier)
    # we assume canonical sample_coord is [V, N, 1, 2]
    if sample_coord.dim() == 4 and sample_coord.shape[1] == N:
        sample_coord = sample_coord[:, idx, :, :].contiguous()
    elif sample_coord.dim() == 4 and sample_coord.shape[0] == N:
        # odd ordering [N, V, 1, 2] and then permute to [V, N,1,2]
        sample_coord = sample_coord.permute(1, 0, 2, 3)[:, idx, :, :].contiguous()
    else:
        raise RuntimeError(f"Unexpected sample_coord shape after canonicalization: {tuple(sample_coord.shape)}")

    if prev_sample_coord.dim() == 4 and prev_sample_coord.shape[1] == M:
        prev_sample_coord = prev_sample_coord[:, prev_idx, :, :].contiguous()
    elif prev_sample_coord.dim() == 4 and prev_sample_coord.shape[0] == M:
        prev_sample_coord = prev_sample_coord.permute(1, 0, 2, 3)[:, prev_idx, :, :].contiguous()
    else:
        raise RuntimeError(f"Unexpected prev_sample_coord shape after canonicalization: {tuple(prev_sample_coord.shape)}")

    # pts_world/prev_pts_world on device and subsampled
    pts_world = pts_world.to(device)[idx].contiguous()
    prev_pts_world = prev_pts_world.to(device)[prev_idx].contiguous()

    # pts_view/prev_pts_view: assume shape [N, V, 3] and [N_prev, V, 3]
    if pts_view is not None:
        pts_view = pts_view.to(device)[idx].contiguous()
    if prev_pts_view is not None:
        prev_pts_view = prev_pts_view.to(device)[prev_idx].contiguous()

    # flow_view shape should be [N, V, 3]
    if flow_view is not None:
        # try to rectify dims: if flow_view was [N, V, 3] -> ok
        # if flow_view is [N, something, 3] where something != V, it's likely malformed
        if flow_view.shape[0] == 1:
            flow_view = flow_view.squeeze(0)
        if flow_view.shape[0] == M:
            flow_view = flow_view[prev_idx.to(device='cpu')].to(device).contiguous()
        elif flow_view.shape[1] == M:
            # shape like [V, N, 1, 3] ? try permute
            flow_view = flow_view.permute(1, 0, 2)[prev_idx.to(device='cpu')].to(device).contiguous()
        else:
            # last resort: try to reshape to [N, V, 3]
            try:
                flow_view = flow_view.reshape(M, -1, 3)[prev_idx].to(device).contiguous()
            except Exception as e:
                raise RuntimeError(f"Can't canonicalize flow_view of shape {tuple(flow_view.shape)}: {e}")

     # flow_view shape should be [N, V, 3]
    if real_flow_view is not None:
        print(real_flow_view.shape)
        # try to rectify dims: if flow_view was [N, V, 3]
        # if flow_view is [N, something, 3] where something != V, its likely malformed
        if real_flow_view.shape[0] == 1:
            real_flow_view = real_flow_view.squeeze(0)
        if real_flow_view.shape[0] == N:
            real_flow_view = real_flow_view[idx.to(device='cpu')].to(device).contiguous()
        elif real_flow_view.shape[1] == N:
            # shape like [V, N, 1, 3] ? try permute
            real_flow_view = real_flow_view.permute(1, 0, 2)[idx.to(device='cpu')].to(device).contiguous()
        else:
            # last resort: try to reshape to [N, V, 3]
            try:
                real_flow_view = real_flow_view.reshape(N, -1, 3)[idx].to(device).contiguous()
            except Exception as e:
                raise RuntimeError(f"Can't canonicalize flow_view of shape {tuple(real_flow_view.shape)}: {e}")


    # gt_prev_pos
    gt_prev_pos = gt_prev_pos.to(device)[idx].contiguous()
    if gt_conn_idx is not None:
        gt_conn_idx = gt_conn_idx.to(device)[idx].contiguous()

    # return a cleaned dict
    return dict(
        sample_coord=sample_coord,           # [V, n_sub, 1, 2]
        pts_world=pts_world,                 # [n_sub, 3]
        pts_view=pts_view,                   # [n_sub, V, 3]
        flow_view=flow_view,                 # [n_sub, V, 3]
        real_flow_view = real_flow_view,
        prev_sample_coord=prev_sample_coord, # [V, m_sub, 1, 2]
        prev_pts_world=prev_pts_world,       # [m_sub, 3]
        prev_pts_view=prev_pts_view,         # [m_sub, V, 3]
        gt_prev_pos=gt_prev_pos,             # [n_sub, 3]
        gt_conn_idx=gt_conn_idx
    )

import math
import torch

def canonicalize_and_subsample_iter(batch, device, num_points:int=1000, num_prev_points:int=1000):
    """
    Yield subsampled chunks covering all points in `batch`.
    Each yielded dict is identical in format to canonicalize_and_subsample(..., num_points=...).

    Notes:
      - If num_points <= 0 -> yields single chunk with all current points
      - Otherwise yields ceil(N/num_points) chunks covering indices [0..N-1]
      - prev points are chunked in parallel (if M differs, prev chunks are iterated with clamping)
    """
    # First canonicalize with "no subsampling" to get full tensors
    canonical_full = canonicalize_and_subsample(batch, device, num_points=0, num_prev_points=0)

    # extract full tensors
    sample_coord_full = canonical_full['sample_coord']       # [V, N, 1, 2]
    prev_sample_coord_full = canonical_full['prev_sample_coord']  # [V, M, 1, 2]
    pts_world_full = canonical_full['pts_world']             # [N, 3]
    prev_pts_world_full = canonical_full['prev_pts_world']   # [M, 3]
    pts_view_full = canonical_full.get('pts_view', None)     # [N, V, 3] or None
    prev_pts_view_full = canonical_full.get('prev_pts_view', None)  # [M, V, 3] or None
    flow_view_full = canonical_full.get('flow_view', None)   # maybe [N, V, 3] or [M, V, 3] depending on upstream
    real_flow_view_full = canonical_full.get('real_flow_view', None)   # expected [N, V, 3] or similar
    gt_prev_pos_full = canonical_full.get('gt_prev_pos', None) # [N,3] or None
    gt_conn_idx_full = canonical_full.get('gt_conn_idx', None)

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
    gt_conn_idx_full = to_device(gt_conn_idx_full)

    N = pts_world_full.shape[0]
    M = prev_pts_world_full.shape[0]

    # if num_points yield once with everything
    if num_points <= 0 or N <= num_points:
        yield dict(
            sample_coord=sample_coord_full.contiguous(),
            pts_world=pts_world_full.contiguous(),
            pts_view=pts_view_full.contiguous() if pts_view_full is not None else None,
            flow_view=flow_view_full.contiguous() if flow_view_full is not None else None,
            real_flow_view=real_flow_view_full.contiguous() if real_flow_view_full is not None else None,
            prev_sample_coord=prev_sample_coord_full.contiguous(),
            prev_pts_world=prev_pts_world_full.contiguous(),
            prev_pts_view=prev_pts_view_full.contiguous() if prev_pts_view_full is not None else None,
            gt_prev_pos=gt_prev_pos_full.contiguous() if gt_prev_pos_full is not None else None,
            gt_conn_idx=gt_conn_idx_full.contiguous() if gt_conn_idx_full is not None else None
        )
        return

    # compute number of chunks (use ceil so we cover remainder)
    num_chunks = int(math.ceil(N / float(num_points)))
    prev_num_chunks = int(math.ceil(M / float(num_prev_points))) if num_prev_points > 0 else 1

    # helper to safe-index along first dim:
    def safe_index(tensor, idx_tensor):
        """
        Index tensor along its first dimension with idx_tensor.
        Moves idx_tensor to tensor.device before indexing.
        """
        if tensor is None:
            return None
        if not isinstance(idx_tensor, torch.Tensor):
            idx_tensor = torch.tensor(idx_tensor, dtype=torch.long, device=tensor.device)
        else:
            idx_tensor = idx_tensor.to(tensor.device)
        # If first dim smaller than max(idx_tensor) -> that is an error; raise clear message
        if idx_tensor.numel() > 0 and idx_tensor.max().item() >= tensor.shape[0]:
            raise RuntimeError(f"Index out of range: trying to index size {tensor.shape[0]} with max index {int(idx_tensor.max().item())}")
        return tensor[idx_tensor].contiguous()

    for i in range(num_chunks):
        start = i * num_points
        end = min((i + 1) * num_points, N)
        idx = torch.arange(start, end, dtype=torch.long, device=device)

        k = 1000
        sub_pts_world = pts_world_full[idx]  # [num_points, 3]
        dists = torch.cdist(sub_pts_world, prev_pts_world_full, p=2)  # [num_points, M]
        _, knn_idx = torch.topk(dists, k=k, dim=1, largest=False)  # [num_points, k]
        prev_idx = knn_idx.reshape(-1)

        if prev_idx.numel() < num_points * k:
            pad_needed = num_points * k - prev_idx.numel()
            pad = prev_idx[:pad_needed % prev_idx.numel()]
            prev_idx = torch.cat([prev_idx, pad], dim=0)

        # sample current and prev as before
        sample_coord = sample_coord_full[:, idx, :, :].contiguous()
        pts_world = pts_world_full[idx].contiguous()
        pts_view = safe_index(pts_view_full, idx)
        gt_prev_pos = safe_index(gt_prev_pos_full, idx)
        gt_conn_idx = safe_index(gt_conn_idx_full, idx)
        
        # pts_world and related -> first dim expected N
        pts_world = safe_index(pts_world_full, idx)
        pts_view = safe_index(pts_view_full, idx) if pts_view_full is not None else None
        gt_prev_pos = safe_index(gt_prev_pos_full, idx) if gt_prev_pos_full is not None else None
        gt_conn_idx = safe_index(gt_conn_idx_full, idx) if gt_conn_idx_full is not None else None

        # real_flow_view is usually aligned with current points (N) but be defensive:
        if real_flow_view_full is None:
            real_flow_view = None
        else:
            # choose which index to use depending on length
            if real_flow_view_full.shape[0] == N:
                real_flow_view = safe_index(real_flow_view_full, idx)
            else:
                # try to reshape first: many times flow is (N, V, 3) but if not, attempt reshape
                try:
                    maybe = real_flow_view_full.reshape(N, -1, real_flow_view_full.shape[-1])
                    real_flow_view = safe_index(maybe, idx)
                except Exception as e:
                    raise RuntimeError(f"Can't canonicalize real_flow_view of shape {tuple(real_flow_view_full.shape)}: {e}")

        # flow_view is often aligned to prev points (M) -- handle similarly
        if flow_view_full is None:
            flow_view = None
        else:
            if flow_view_full.shape[0] == M:
                flow_view = safe_index(flow_view_full, prev_idx)
            else:
                try:
                    maybe = flow_view_full.reshape(M, -1, flow_view_full.shape[-1])
                    flow_view = safe_index(maybe, prev_idx)
                except Exception as e:
                    raise RuntimeError(f"Can't canonicalize flow_view of shape {tuple(flow_view_full.shape)}: {e}")

        # prev-frame items
        prev_sample_coord = prev_sample_coord_full[:, prev_idx, :, :].contiguous() if prev_sample_coord_full is not None else None
        prev_pts_world = safe_index(prev_pts_world_full, prev_idx)
        prev_pts_view = safe_index(prev_pts_view_full, prev_idx) if prev_pts_view_full is not None else None

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
            gt_conn_idx=gt_conn_idx
        )

