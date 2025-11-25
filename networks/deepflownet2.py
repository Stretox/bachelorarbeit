import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.UnetSimple import UNetSimple

# 🐳 <- Cute (And the best emoji in unicode)

from einops import rearrange


# Positional encoding from nerf-pytorch
# Embedder and get_embedder as well as Transformer functions by MonoHair: https://github.com/KeyuWu-CS/MonoHair?tab=License-1-ov-file
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

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim=64, dir_lambda=0.5, dropout = 0.):
        super().__init__()
        self.norman = nn.LayerNorm(dim)
        self.attn = DirectionalCrossAttention(dim, heads = heads, dim_head = dim_head, dir_lambda=dir_lambda, dropout = dropout)
        self.ff = Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
    def forward(self, x_q, x_kv, pos_q, pos_kv, prev_flow_dir):
        x_q = self.norman(x_q)
        x = self.attn(x_q, x_kv, pos_q, pos_kv, prev_flow_dir)
        x += x_q
        x = self.ff(x)
        return x

################################################################################################################################################

class DirectionalCrossAttention(nn.Module):
    """
    Returns:
        out: [N, C] attended outputs
        attn_weights: [N, M] attention probabilities (rows sum to 1)
    
    """
    def __init__(self, dim, heads=8, dim_head=64, dir_lambda=1.0, dropout=0.):

        super().__init__()
        
        inner = dim_head * heads # Computes the inner dimension
        self.heads = heads 
        self.scale = dim_head ** -0.5 # 1/sqrt of the head dimension (eg, 8)
        # linear layer: every input connected to every output
        self.to_q = nn.Linear(dim, inner, bias=False) # linear Layer for Query
        self.to_kv = nn.Linear(dim, inner * 2, bias=False) # linear layer for key/value (notice that inner is now 2*inner)
        self.Wo = nn.Sequential(
            nn.Linear(inner, dim),
            nn.Dropout(dropout) # Dropout like MonoHair
        ) # linear layer for output
        self.dir_lambda = dir_lambda

    # Input: ((self), cur_tokens, prev_tokens, pts_world, prev_pts_world, prev_flow_dir)
    # Where flow direction is the movement of the previous points!!!
    def forward(self, x_q, x_kv, pos_q, pos_kv, prev_flow_dir):
        """
        Computes the Cross Attention forward when calling the class
        
        Inputs:

        x_q: Query embeddings (current points)
        x_kv: Key/value embeddings (previous points)
        pos_q: World coordinates of current points
        pos_kv: World coordinates of previous points
        prev_flow_dir: flow direction vectors for prev_points
        """

        # q, k and v are projected to the inner dimension using the Linear Layers

        q = self.to_q(x_q) # [N, inner]
        k, v = self.to_kv(x_kv).chunk(2, dim=-1)  # [M, inner] for both

        # get the d
        d = q.shape[-1] // self.heads
        assert q.shape[-1] == self.heads * d, "Feature dimension must be divisible by number of heads"
        
        # Reshaping tensors to [heads, N, dim_head] for parallel head attention
        q = rearrange(q, 'n (h d) -> h n d', h=self.heads) # q now [h,n,d]
        k = rearrange(k, 'm (h d) -> h m d', h=self.heads) # k now [h,m,d]
        v = rearrange(v, 'm (h d) -> h m d', h=self.heads) # v now [h,m,d]

        L = torch.einsum('h n d, h m d -> h n m', q, k) * self.scale # [h, N, M]
        # L = L.clamp(min=-50, max=50) #DEBUG CLAMPING

        # compute direction vectors from query to each prev_point and use as alignment bias
        delta = pos_kv.unsqueeze(0) - pos_q.unsqueeze(1) # [N, M, 3]

        delta_norm = torch.norm(delta, dim=-1, keepdim=True).clamp(min=1e-6)  # [N, M, 1]
        delta_unit = delta / delta_norm

        # normalize flow_dir
        prev_flow_unit = F.normalize(prev_flow_dir, dim=-1)  # [N,3]

        # compute per-query-point alignment using cosinus formula
        align = torch.einsum('nd,nmd->nm', prev_flow_unit, delta_unit.transpose(0,1)).transpose(0,1)  # values are in [-1,1] depending on the alignment

        # TODO: Possibly Overfiting keep lambda small
        # Applying the directional bias
        L = L + (self.dir_lambda * align).unsqueeze(0) # (unsqueeze to apply to all attention heads)

        # softmax
        W_h = torch.softmax(L, dim=-1)  # [h, N, M]
        
        # Now we finally multiply the values
        O = torch.einsum('h n m, h m d -> h n d', W_h, v) # W_hV
        
        # merge heads
        O = rearrange(O, 'h n d -> n (h d)') # [N, inner]
        O = self.Wo(O)  # [N, C]
        return O

# using DirectionalCrossAttention we try to predict point correspondences and motion between two sets of 3D points
class DeepFlowNet(nn.Module):
    
    def __init__(self, in_feat=3, token_dim=128, vit_heads=8, num_views=32, pt_res=5,
                dir_lambda=2.0, debug=False, training=True):
        
        # Debug ein und ausschalten
        self.debug = debug

        # Training or Not
        self.training = training

        super().__init__()
        # backbone for deep feature maps
        self.backbone = UNetSimple(in_channels=in_feat, ksize=5)
        
        # positional point embedding
        self.pt_embed, self.pt_dim = get_embedder(pt_res, input_dims=3)

        # used for
        self.view_fuse = nn.Linear(self.backbone.output_feat + self.pt_dim, token_dim)

        # Cross-attention block with motion bias
        self.cross_attn = DirectionalCrossAttention(dim=token_dim, heads=vit_heads, dim_head=token_dim // vit_heads, dir_lambda=dir_lambda)

        self.cross_attn2 = Transformer(dim=token_dim, heads=vit_heads, dim_head=token_dim // vit_heads, dir_lambda=dir_lambda)

        # small MLP to predict motion residuals aka a simple decoder
        # predicts: prev_pos_pred - current_pos
        # More Complex like Monohair now
        self.MLP = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, 3) 
        )

        self.fuse_mlp = nn.Sequential(
            nn.LayerNorm((num_views+1)*token_dim),
            nn.Linear((num_views+1)*token_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim)
        ) # MLP but smaller. Fuses per view features to per point features

        self.fuse_tokens_over_views = nn.Linear(token_dim*num_views, token_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_views + 1, token_dim))

        self.num_views = num_views

        # losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.mse = nn.MSELoss()

        # Class and View Tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        self.cls_fuse_pt = nn.Linear(token_dim + self.pt_dim, token_dim)

    def sample_feats_from_backbone(self, imgs, sample_coord):
        """
        wrapper function
        
        returns features with per view point embeddings and permutes them to [N, V, C_f]
        """

        # backbone returns with shape [V, C_f, N]
        feat = self.backbone(imgs, sample_coord).permute(2, 0, 1) # permute to [N, V, C_f]
        return feat

    def fuse_per_point_token(self, img_feat, pts_view_feat, pts_world_feat):
        """
        Input 
        img_feat: The Image features from the backbone | shape: [N, V, C_f]
        pts_view_feat The per point embeddings for each point | shape: [N, V, C_pt]
        
        returns: per-point token embeddings | shape: [N, token_dim]
        """
        
        inp = torch.cat([img_feat, pts_view_feat], dim=-1)  # [N, V, C_f + C_pt]

        # map each view to token_dim
        view_tokens = self.view_fuse(inp) # [N, V, token_dim]

        n = view_tokens.shape[0] # N
        cls_tokens = self.cls_token.expand(n, -1, -1) # [N, 1, token_dim]

        cls_tokens = self.cls_fuse_pt(torch.cat([cls_tokens, pts_world_feat.unsqueeze(1)], dim=-1))

        fused = torch.cat((cls_tokens, view_tokens), dim=1)  # [N, V+1, token_dim]
        
        fused += self.pos_embedding

        fused_flat = fused.reshape(n, -1) 

        # use mean to fuse over views
        point_tokens = self.fuse_mlp(fused_flat)  # [N, token_dim]
        return point_tokens

    def forward(self, data):
        """
        Main Model forward. Returns a dict with:

        'W': attention weights | shape [N, N_prev]
        'prev_pos_pred': weighted prev position | shape [N, 3]
        'motion_res': predicted residual | shape [N, 3]
        
        !!! and the losses for backward !!!

        """

        # data (dict(input)) :
        # ['imgs']
        # ['pts_world']
        # ['pts_view']
        # ['prev_feats']
        # ['prev_pts_world']
        # ['gt_prev_pos']
        # ...

        ############################## get current per-point features ########################################
        imgs = data['imgs']                # [V, C, H, W]
        # masks = data.get('masks', None) # backbone doesn't use masks atm
        sample_coord = data['sample_coord']  # [V, N, 1, 2]
        pts_world = data['pts_world']        # [N, 3]
        pts_view = data.get('pts_view', None) # [N, V, 3]


        # flow per view: your flow direction per view projected to the 3D query
        #TODO: NOT Optional Optional Optional Optional Optional Optional Optional Optional Optional
        prev_flow = data.get('flow_view', None)  # [N, 3] 
        prev_flow_dir = F.normalize(prev_flow, dim=-1)  # [N, 3]

        # GT flow direction
        real_flow_view = data.get('real_flow_view', None)

        # print(f"1 {real_flow_view.shape}")

        real_flow_dir = F.normalize(real_flow_view, dim=-1)

        # print(f"2 {real_flow_dir.shape}")

        #################### per point features ########################################################

        # extract current points features (deep feature maps)
        img_feat = self.sample_feats_from_backbone(imgs, sample_coord) # (N, V, C_f)

        # Embed world coordinates
        pts_world_feat = self.pt_embed(pts_world)
    
        # flatten per-view pts_view into per-view embeddings
        # pts_view points in camera view | shape [N, V, 3]
        # reshape to [N*V,3], embed, then reshape back

        N = pts_view.shape[0]; V = pts_view.shape[1]
        pv_flat = pts_view.reshape(-1, 3)
        pv_emb = self.pt_embed(pv_flat).reshape(N, V, -1)
        pts_view_feat = pv_emb

        # per-point token generation
        cur_tokens = self.fuse_per_point_token(img_feat, pts_view_feat=pts_view_feat, pts_world_feat=pts_world_feat)  # Returned: [N, token_dim]

        ##################### get previous per-point features ############################################

        # per view features
        prev_img_feat = self.sample_feats_from_backbone(data['prev_imgs'], data['prev_sample_coord'])

        prev_pts_world = data['prev_pts_world']  # [N_prev, 3]
        prev_pts_view = data['prev_pts_view'] # 

        # Embed world coordinates
        prev_pts_world_feat = self.pt_embed(prev_pts_world)
        
        N = prev_pts_view.shape[0]
        V = prev_pts_view.shape[1]
        ppv_flat = prev_pts_view.reshape(-1, 3)
        ppv_emb = self.pt_embed(ppv_flat).reshape(N, V, -1)
        prev_pts_view_feat = ppv_emb

        prev_tokens = self.fuse_per_point_token(prev_img_feat, pts_view_feat=prev_pts_view_feat, pts_world_feat=prev_pts_world_feat)  # [N_prev, token_dim]

        ####################### cross-attention: current tokens attend to prev tokens ############################################
        
        # DirectionalCrossAttention input:
        # embeddings + pts_world.shape = [N,3], prev_pts_world.shape = [N_prev,3], prev_flow_dir.shape = [N,3]
        O = self.cross_attn2(cur_tokens, prev_tokens, pts_world, prev_pts_world, prev_flow_dir)

        # predicted previous position: weighted sum of previous points
        # prev_pos_pred = torch.matmul(W, prev_pts_world)  # [N, 3] (same size as pts_world)

        # Predicted motion with MLP
        pred_points = self.MLP(O)  # [N, 3] (predicted points)
        
        # position from motion_dir for loss
        motion_dir = pts_world - pred_points

        out = {
            # 'W': W, # [N, N_prev] indexes (attention output)
            # 'prev_pos_pred': prev_pos_pred, # attention-weighted prev pos estimation
            'motion_dir': motion_dir, # Motion per point (kinda normalized)
            'pred_points': pred_points
        }

        ####################### losses ####################################################### (only for training obvs)
        losses = {}

        if self.training:

            gt_prev_pos  = data['gt_prev_pos'] # [N, 3]

            mse = self.mse(pred_points, gt_prev_pos)

            losses['mse'] = mse

            # flow consistency (predicted prev position shall be be along real flow dir)
            vec = F.normalize(motion_dir, dim=-1)  # direction from current to predicted previous position
            flow_u = F.normalize(real_flow_dir, dim=-1) # Normalized
            cos = (flow_u * vec).sum(dim=-1) # Cosine similarity
            flow_consistency = torch.mean(1.0 - cos) # Loss: 1 - cos
            losses['flow_consistency'] = flow_consistency # to output

            out['losses'] = losses
        return out