import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.UnetSimple import UNetSimple

# Here is a whale to help you get through this mess of a code:
# 🐳
#whalesmakeeverythingbetter

# Positional encoding from nerf-pytorch
# Embedder and get_embedder by MonoHair: https://github.com/KeyuWu-CS/MonoHair?tab=License-1-ov-file
# under Attribution-NonCommercial 4.0 International
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

################################################################################################################################################

from einops import rearrange

class DirectionalCrossAttention(nn.Module):
    """
    Returns:
      out: [N, C] attended outputs
      attn_weights: [N, M] attention probabilities (rows sum to 1)
    """
    def __init__(self, dim, heads=8, dim_head=64, dir_lambda=1.0):


        super().__init__()
        
        inner = dim_head * heads # Computes the inner dimension
        self.heads = heads 
        self.scale = dim_head ** -0.5 # 1/sqrt of the head dimension (eg, 8)
        # linear layer: every input connected to every output
        self.to_q = nn.Linear(dim, inner, bias=False) # linear Layer for Query
        self.to_kv = nn.Linear(dim, inner * 2, bias=False) # linear layer for key/value (notice that inner is now 2*inner)
        self.to_out = nn.Linear(inner, dim) # linear layer for output
        self.dir_lambda = dir_lambda

    # ((self), cur_tokens, prev_tokens, pts_world, prev_pts_world, flow_dir)
    # Where flow direction is the movement of the previous points
    def forward(self, x_q, x_kv, pos_q, pos_kv, flow_dir=None):
        """
        Inputs:

        - x_q: Query embeddings (current points)
        - x_kv: Key/value embeddings (previous points)
        - pos_q: World coordinates of current points
        - pos_kv: World coordinates of previous points
        - flow_dir: Unit flow direction vectors (optional)
        """

        # x_q: [N, C]; x_kv: [M, C]
        # Handle both [N, C] and [N, V, C] inputs
        is_3d = x_q.dim() == 3
        if is_3d:
            # Flatten views into batch: [N*V, C]
            N, V, C = x_q.shape
            M, _, _ = x_kv.shape
            x_q = x_q.reshape(N*V, C)
            x_kv = x_kv.reshape(M*V, C)
            # Repeat pos and flow_dir for each view
            pos_q = pos_q.unsqueeze(1).expand(-1, V, -1).reshape(N*V, 3)
            pos_kv = pos_kv.unsqueeze(1).expand(-1, V, -1).reshape(M*V, 3)
            if flow_dir is not None:
                flow_dir = flow_dir.unsqueeze(1).expand(-1, V, -1).reshape(N*V, 3)
        else:
            N, C = x_q.shape
            M, _ = x_kv.shape

        # q, k, v are projected to the inner dimension.
        q = self.to_q(x_q) # [N, inner]
        k, v = self.to_kv(x_kv).chunk(2, dim=-1)  # [M, inner] each

        # reshape for multihead: [h, N, d] / [h, M, d]
        d = q.shape[-1] // self.heads # don't forget to use #attentionisallyouneed
        assert q.shape[-1] == self.heads * d, "Feature dimension must be divisible by number of heads"
        # Reshapes tensors to [heads, N, dim_head] for parallel processing.
        
        q = rearrange(q, 'n (h d) -> h n d', h=self.heads) # q now [h,n,d]
        k = rearrange(k, 'm (h d) -> h m d', h=self.heads) # k now [h,m,d]
        v = rearrange(v, 'm (h d) -> h m d', h=self.heads) # v now [h,m,d]

        # content logits: [h, N, M]
        # Computes dot-product attention scores, scaled.
        logits = torch.einsum('h n d, h m d -> h n m', q, k) * self.scale
        # logits = logits.clamp(min=-50, max=50) #TODO: CLAMP FOR DEBUG

        # if a flow_dir is provided compute a directional alignment bias
        if flow_dir is not None:
            # compute direction vectors from query to each prev point: [N, M, 3]
            # pos_q: [N,3], pos_kv: [M,3]
            # delta[n,m,3] = pos_kv[m] - pos_q[n]
            # normalize delta to unit vectors (handle zero distances)
            delta = pos_kv.unsqueeze(0) - pos_q.unsqueeze(1)            # [1, M, 3] - [N,1,3] = [N, M, 3]
            # now delta: [N, M, 3]
            delta_norm = torch.norm(delta, dim=-1, keepdim=True).clamp(min=1e-6)  # [N, M, 1]
            delta_unit = delta / delta_norm

            # normalize flow_dir [N,3]
            flow_unit = F.normalize(flow_dir, dim=-1)  # [N,3]

            # cos alignment: [N, M] = flow_unit[n] dot delta_unit[n,m]
            # compute per-query alignment
            dir_align = torch.einsum('nd,nmd->nm', flow_unit, delta_unit)  # values in [-1,1]

            # broadcast to heads and scale by lambda
            # logits' shape is [h, N, M], so unsqueeze(0) into heads
            # TODO: Possibly Explosive
            logits = logits + (self.dir_lambda * dir_align).unsqueeze(0)

        # softmax over previous points (last dim)
        attn = torch.softmax(logits, dim=-1)  # [h, N, M]
        # aggregate values: [h, N, d] = attn @ v
        out = torch.einsum('h n m, h m d -> h n d', attn, v)
        # merge heads: [N, inner]
        out = rearrange(out, 'h n d -> n (h d)')
        out = self.to_out(out)  # [N, C]
        # combine attention across heads (average) to produce [N, M] connection probabilities
        attn_weights = attn.mean(dim=0)  # [N, M]
        return out, attn_weights

# This class uses the Class DirectionalCrossAttention to predict point correspondences and motion between two sets of 3D points

class MovementAttn(nn.Module):
    
    def __init__(self, in_feat=3, token_dim=128, vit_heads=8, num_views=4, pt_res=5,
                dir_lambda=2.0, use_pos=True, use_pt=True):
        

        super().__init__()
        # backbone (same design as your Ori/Occ models)
        self.backbone = UNetSimple(in_channels=in_feat, ksize=5)
        
        self.base_img_ch = in_feat   # store base image channels

        # Sets up positional embedding (pt_embed)
        self.pt_embed, self.pt_dim = get_embedder(pt_res, input_dims=3)

        # a small linear to convert view-fused -> token_dim
        self.view_fuse = nn.Linear(self.backbone.output_feat + self.pt_dim if use_pt else self.backbone.output_feat, token_dim)

        # Cross-attention blockk
        self.cross_attn = DirectionalCrossAttention(dim=token_dim, heads=vit_heads, dim_head=token_dim // vit_heads, dir_lambda=dir_lambda)

        # optionally a small decoder to predict motion residuals (if you also want regression)
        self.motion_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, 3)   # predicts residual: prev_pos_pred - current_pos
        )

        self.use_pt = use_pt
        self.num_views = num_views
        self.use_pos = use_pos

        # losses that might be used externally
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()

    def sample_feats_from_backbone(self, imgs, sample_coord):
        """
        puts 
        Helper: returns [N, V, C_f]
        """
        # backbone returns [V, C_f, N] and permute to [N, V, C_f]
        feat = self.backbone(imgs, sample_coord).permute(2, 0, 1)
        return feat

    def fuse_per_point_token(self, img_feat, pts_view_feat=None, pts_world_feat=None):
        """
        img_feat: [N, V, C_f]
        pts_view_feat: [N, V, C_pt] or None
        returns: per-point token embeddings [N, token_dim]
        """
        if self.use_pt and pts_view_feat is not None:
            inp = torch.cat([img_feat, pts_view_feat], dim=-1)  # [N, V, C_f + C_pt]
        else:
            inp = img_feat  # [N, V, C_f]
        # pts_world_feat
        # map each view to token_dim
        view_tokens = self.view_fuse(inp)  # [N, V, token_dim]
        # aggregate across views -> we use mean as a simple per-point embedding
        point_tokens = view_tokens.mean(dim=1)  # [N, token_dim] # UNUSED
        return point_tokens, view_tokens

    def forward(self, data):
        """
        Main Model forward. Returns a dict with:
          - 'alpha': [N, N_prev] attention weights
          - 'prev_pos_pred': [N, 3] weighted prev position
          - 'motion_res': [N, 3] predicted residual (optional)
          - the losses
        All just useful for interpreting
        """

        # data (dict) :
        # ['imgs']
        # ['masks']
        # ['pts_world']
        # ['pts_view']
        # ['prev_feats']
        # ['prev_pts_world']
        # ['gt_prev_pos']
        # ...

        ############################## get current per-point features ########################################
        imgs = data['imgs']                # [V, C, H, W]
        # masks = data.get('masks', None) # Masks would be better but eh
        sample_coord = data['sample_coord']  # [V, N, 1, 2]
        pts_world = data['pts_world']        # [N, 3]
        pts_view = data.get('pts_view', None) # [N, V, 3] (optional)


        # flow per view: your flow direction per view projected to the 3D query
        #TODO: NOT Optional Optional Optional Optional Optional Optional Optional Optional Optional
        flow_view = data.get('flow_view', None)  # [N, V, 3] (per-view direction estimate)
        if flow_view is not None:
            # average and normalize flow directions across views that have valid masks
            flow_dir = F.normalize(flow_view.mean(dim=1), dim=-1)  # [N, 3]
        else:
            flow_dir = None

        # GT flow direction
        real_flow_view = data.get('real_flow_view', None)

        # print(f"1 {real_flow_view.shape}")

        real_flow_dir = F.normalize(real_flow_view.mean(dim=1), dim=-1)

        # print(f"2 {real_flow_dir.shape}")

        # extract current points features (deep feature maps)
        img_feat = self.sample_feats_from_backbone(imgs, sample_coord) # (N, V, C_f)

        # Embed world coordinates
        pts_world_feat = self.pt_embed(pts_world) if self.use_pt else None
    
        # flatten per-view pts_view into per-view embeddings
        # pts_view shape [N, V, 3] | reshape to [N*V,3], embed, then reshape back
        N = pts_view.shape[0]; V = pts_view.shape[1]
        pv_flat = pts_view.reshape(-1, 3)
        pv_emb = self.pt_embed(pv_flat).reshape(N, V, -1)
        pts_view_feat = pv_emb

        # Aggregates features across views into per-point tokens.
        cur_tokens, _ = self.fuse_per_point_token(img_feat, pts_view_feat=pts_view_feat, pts_world_feat=pts_world_feat)  # [N, token_dim]

        ##################### get previous per-point features ############################################

        # if prev_feats in data and data['prev_feats'] is not None:
        #     prev_img_feat = data['prev_feats']  # expected [N_prev, V, C_f]
        # elif 'prev_sample_coord' in data and 'prev_imgs' in data:
        if 'prev_sample_coord' in data and 'prev_imgs' in data:
            prev_img_feat = self.sample_feats_from_backbone(data['prev_imgs'], data['prev_sample_coord'])
        else:
            raise ValueError('Please provide prev_feats or prev_sample_coord+prev_imgs')

        prev_pts_world = data['prev_pts_world']  # [N_prev, 3]
        prev_pts_view = data.get('prev_pts_view', None)
        prev_pts_world_feat = self.pt_embed(prev_pts_world) if self.use_pt else None
        
        Np = prev_pts_view.shape[0]; V = prev_pts_view.shape[1]
        ppv_flat = prev_pts_view.reshape(-1, 3)
        ppv_emb = self.pt_embed(ppv_flat).reshape(Np, V, -1)
        prev_pts_view_feat = ppv_emb

        prev_tokens, _ = self.fuse_per_point_token(prev_img_feat, pts_view_feat=prev_pts_view_feat, pts_world_feat=prev_pts_world_feat)  # [N_prev, token_dim]

        ####################### cross-attention: current tokens attend to prev tokens ############################################
        # DirectionalCrossAttention input:
        # pos_q = pts_world [N,3], pos_kv = prev_pts_world [N_prev,3], flow_dir aggregated [N,3]
        attn_out, alpha = self.cross_attn(cur_tokens, prev_tokens, pts_world, prev_pts_world, flow_dir)

        # predicted previous position: weighted sum of previous points
        prev_pos_pred = torch.matmul(alpha, prev_pts_world)  # [N, 3] (same size as pts_world)

        # Predicted motion #    residual
        motion_res = self.motion_head(attn_out)  # [N, 3] (Motion Vectors for )
        # interpret as predicted_prev_pos = current_pos + motion_res
        prev_pos_pred_from_res = pts_world + motion_res

        out = {
            'alpha': alpha,                       # [N, N_prev] indexes
            'prev_pos_pred': prev_pos_pred,       # attention-weighted prev pos estimation/prediction. Married to alpha
            'motion_res': motion_res,             # regressed residual. Direct mapping to a prev point
            'prev_pos_pred_from_res': prev_pos_pred_from_res # Vector pointing along motion_res
        }

        ####################### losses if GT provided ################################ (For training obvs)
        losses = {}

        print(alpha.requires_grad)
        print(alpha[0, :10])

        # Soft supervision for attention weights using 3D distance
        if 'pts_world' in data and 'gt_prev_pos' in data and 'prev_pts_world' in data:
            query_pos_gt = data['pts_world']       # [N, 3]
            gt_prev_pos  = data['gt_prev_pos']     # [N, 3]
            prev_pos_gt  = data['prev_pts_world']  # [N_prev, 3]

            # compute distances between each gt_prev_pos (1 per query) and all candidate prev points
            with torch.no_grad():
                dist = torch.cdist(gt_prev_pos, prev_pos_gt)  # [N, N_prev]
                print(dist.min(dim=-1).values.mean())

                # use adaptive temperature for stable softmax sharpness
                temp = max(dist.median().item() * 0.3, 1e-3)  # e.g. median*0.5
                soft_target = F.softmax(-dist / temp, dim=-1)

            print(soft_target[0].max(), soft_target[0].mean())

            # shape check
            assert soft_target.shape == alpha.shape, f"soft_target={soft_target.shape}, alpha={alpha.shape}"

            # stable log and soft cross-entropy
            log_alpha = (alpha.clamp(min=1e-6)).log()
            soft_ce = -(soft_target * log_alpha).sum(dim=-1).mean()

            losses['conn_ce_soft'] = soft_ce


        # regression loss between attention-weighted prev pos and GT prev pos ## Insert sunglasses emoji meme ##
        if 'gt_prev_pos' in data and data['gt_prev_pos'] is not None:
            gt_prev_pos = data['gt_prev_pos']
            l1 = self.l1_loss(prev_pos_pred, gt_prev_pos)
            l1_res = self.l1_loss(prev_pos_pred_from_res, gt_prev_pos)
            losses['conn_l1_attn'] = l1
            losses['conn_l1_res'] = l1_res
            # Just standard stuff basically

        # flow consistency (encourages predicted prev position to be along real flow dir)
        if flow_dir is not None:
            # vector from current to predicted prev pos
            vec = F.normalize(prev_pos_pred - pts_world, dim=-1)
            flow_u = F.normalize(flow_dir, dim=-1)
            # print(vec.shape, flow_u.shape)
            # cosine similarity (want them to align negatively because flow is movement from prev->cur; if flow indicates cur direction, prev is opposite)
            cos = (flow_u * vec).sum(dim=-1)  # [N]
            # if aligned cos will be closer to 1.0
            # encourage cos to be close to -1 if flow points current->next, but since flow semantics are without any reason (Due to bad flow definition in training data), allow positive reward for alignment
            # here we encourage dot(flow, (prev - cur)) > 0 then prev is in direction opposite to flow #TODO: Flip sign????
            flow_consistency = torch.mean(1.0 - cos)  # smaller when aligned
            losses['flow_consistency'] = flow_consistency

        out['losses'] = losses
        return out
