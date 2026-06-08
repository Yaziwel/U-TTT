import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers 
from torch import einsum
from einops import rearrange
import math 
import torch.distributed as dist
from timm.models.layers import trunc_normal_


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c d h w -> b (d h w) c')

def to_4d(x,d,h,w):
    return rearrange(x, 'b (d h w) c -> b c d h w',d=d,h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        d, h, w = x.shape[-3:]
        return to_4d(self.body(to_3d(x)), d, h, w)





#########################################################################
# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv3d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv3d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=1, bias=bias) 

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = F.gelu(x1) 
        x = x1*x2
        x = self.project_out(x) 

        return x


class SpaTTT3D(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads
        # qkv: 1x1x1 conv3d
        self.qkv = nn.Conv3d(dim, dim * 3 + head_dim * 3, kernel_size=1, bias=bias)
        # depthwise 3x3x3 conv (groups = out_channels -> depthwise)
        out_ch = dim * 3 + head_dim * 3
        self.qkv_dwconv = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=out_ch, bias=bias)

        # inner parameters
        # w1, w2: for simplified swiglu (same shapes as 2D version)
        self.w1 = nn.Parameter(torch.ones(1, self.num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.ones(1, self.num_heads, head_dim, head_dim))
        # w3: now 3D kernel (head_dim, 1, 3, 3, 3)
        self.w3 = nn.Parameter(torch.ones(head_dim, 1, 3, 3, 3))

        trunc_normal_(self.w1, std=.02)
        trunc_normal_(self.w2, std=.02)
        trunc_normal_(self.w3, std=.02)

        # projection: same as before (per-position linear)
        self.proj = nn.Linear(dim + head_dim, dim)

        # scales
        self.scale1 = head_dim ** -0.5
        equivalent_head_dim = 27  # 3x3x3
        self.scale2 = equivalent_head_dim ** -0.5

        # learning-rate-like per-parameter multipliers
        self.swiglu_lr_1 = nn.Parameter(torch.ones(1, num_heads, head_dim, head_dim))
        self.swiglu_lr_2 = nn.Parameter(torch.ones(1, num_heads, head_dim, head_dim))
        self.dwc_lr = nn.Parameter(torch.ones(head_dim, 1, 3, 3, 3))

    def inner_train_simplified_swiglu(self, k, v, w1, w2, lr=1.0):

        # --- Forward ---
        z1 = k @ w1
        z2 = k @ w2
        sig = torch.sigmoid(z2)
        a = z2 * sig

        # --- Backward (analytically derived) ---
        e = - v / float(v.shape[2]) * self.scale1  # avg over positions
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))

        # Clip for stability
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)

        # Gradient step-like update (using learned lr scalars)
        w1 = w1 - self.swiglu_lr_1 * g1
        w2 = w2 - self.swiglu_lr_2 * g2
        return w1, w2

    def inner_train_3x3x3_dwc(self, k, v, w, lr=1.0):

        B, C, D, H, W = k.shape
        # e = dl/dv_hat; average over spatial volume
        e = - v / float(v.shape[2] * v.shape[3] * v.shape[4]) * self.scale2

        # pad k on D,H,W with 1 on both sides
        kp = F.pad(k, (1, 1, 1, 1, 1, 1))  # pad format: (wL,wR,hL,hR,dL,dR)

        outs = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    zs = 1 + dz
                    ys = 1 + dy
                    xs = 1 + dx
                    # slice same-size window and dot with e, sum over spatial dims
                    patch = kp[:, :, zs: zs + D, ys: ys + H, xs: xs + W]
                    # sum over spatial volume -> shape (B, C)
                    dot = (patch * e).sum(dim=(-3, -2, -1))
                    outs.append(dot)
        # outs length = 27, stack into (..., 27) then reshape to (B*C, 1, 3,3,3)
        g = torch.stack(outs, dim=-1).reshape(B * C, 1, 3, 3, 3)

        # Clip 
        norm = torch.sqrt((g ** 2).sum(dim=(-3, -2, -1), keepdim=True) + 1e-6)
        g = g / (norm + 1.0)

        # Step: create per-sample-per-channel updated kernel
        # w shape (head_dim,1,3,3,3) -> repeat B times to get (B*head_dim,1,3,3,3)
        w_rep = w.repeat(B, 1, 1, 1, 1)
        dwc_lr_rep = self.dwc_lr.repeat(B, 1, 1, 1, 1)
        w_new = w_rep - dwc_lr_rep * g
        return w_new

    def forward(self, x):
        """
        x: [B, C, D, H, W]
        """
        b, c, depth, h, w = x.shape
        n = depth * h * w
        d_head = c // self.num_heads

        # qkv conv 3d + depthwise conv3d
        x = self.qkv_dwconv(self.qkv(x))  # shape [B, out_ch, D, H, W]

        # flatten spatial dims to sequence length N
        x = rearrange(x, 'b ch d h w -> b (d h w) ch')

        # split to q1,k1,v1 (per-head attention) and q2,k2,v2 (depthwise 3D conv params)
        q1, k1, v1, q2, k2, v2 = torch.split(x, [c, c, c, d_head, d_head, d_head], dim=-1)

        # prepare q/k/v for the swiglu inner training
        q1 = q1.reshape(b, n, self.num_heads, d_head).transpose(1, 2)   # (b, num_heads, n, d_head)
        k1 = k1.reshape(b, n, self.num_heads, d_head).transpose(1, 2)
        v1 = v1.reshape(b, n, self.num_heads, d_head).transpose(1, 2)

        # prepare q2/k2/v2 as volumetric tensors for depthwise 3D inner training
        # reshape to (b, depth, h, w, d_head) then permute to (b, d_head, depth, h, w)
        q2 = q2.reshape(b, depth, h, w, d_head).permute(0, 4, 1, 2, 3)
        k2 = k2.reshape(b, depth, h, w, d_head).permute(0, 4, 1, 2, 3)
        v2 = v2.reshape(b, depth, h, w, d_head).permute(0, 4, 1, 2, 3)

        # Inner (fast) training
        w1, w2 = self.inner_train_simplified_swiglu(k1, v1, self.w1, self.w2)
        w3 = self.inner_train_3x3x3_dwc(k2, v2, self.w3)

        # Apply updated inner module to q
        # Part1: swiglu-like per-head linear gating
        x1 = (q1 @ w1) * F.silu(q1 @ w2)  # (b, num_heads, n, d_head)
        x1 = x1.transpose(1, 2).reshape(b, n, c)  # (b, n, c)

        # Part2: apply depthwise 3D conv to q2
        # q2 currently (b, d_head, depth, h, w) -> reshape to (1, b*d_head, depth, h, w)
        x2 = F.conv3d(q2.reshape(1, b * d_head, depth, h, w), w3, padding=1, groups=b * d_head)
        # x2 shape: (1, b*d_head, depth, h, w) -> reshape to (b, d_head, n)
        x2 = x2.reshape(b, d_head, n).transpose(1, 2)  # (b, n, d_head)

        # Concat and project
        x = torch.cat([x1, x2], dim=-1)  # (b, n, c + d_head)
        x = self.proj(x)  # (b, n, dim)
        # reshape back to (b, c, d, h, w)
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=depth, h=h, w=w)
        return x 



class FreqTTT3D(nn.Module):

    def __init__(self, dim, num_heads, patch_size=8, bias=False):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        head_dim = dim // num_heads

        # QKV projection (same as original TTT)
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=3, padding=1,
                                    groups=dim * 3, bias=bias)

        # Inner SWIGLU parameters (UNCHANGED SHAPE)
        self.w1 = nn.Parameter(torch.ones(1, num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.ones(1, num_heads, head_dim, head_dim))
        trunc_normal_(self.w1, std=.02)
        trunc_normal_(self.w2, std=.02)

        self.swiglu_lr_1 = nn.Parameter(torch.ones(1, num_heads, head_dim, head_dim))
        self.swiglu_lr_2 = nn.Parameter(torch.ones(1, num_heads, head_dim, head_dim))

        self.proj = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

        self.scale1 = head_dim ** -0.5

    # -------------------------------------------------
    # INNER TTT (unchanged math)
    # -------------------------------------------------
    def inner_train_simplified_swiglu(self, k, v, w1, w2):
        # k,v: [B, H, N, d]
        z1 = k @ w1
        z2 = k @ w2

        sig = torch.sigmoid(z2)
        a = z2 * sig

        e = - v / float(v.shape[2]) * self.scale1

        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))

        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)

        w1 = w1 - self.swiglu_lr_1 * g1
        w2 = w2 - self.swiglu_lr_2 * g2
        return w1, w2

    # -------------------------------------------------
    # FORWARD
    # -------------------------------------------------
    def forward(self, x):
        """
        x: [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        p = self.patch_size

        # ---- QKV conv ----
        x = self.qkv_dwconv(self.qkv(x))  # [B, 3C, D, H, W]

        # ---- Patchify ----
        x_patch = rearrange(
            x, 'b c (d pd) (h ph) (w pw) -> b c d h w pd ph pw',
            pd=p, ph=p, pw=p
        )

        # ---- FFT (3D) ----
        x_fft = torch.fft.rfftn(x_patch.float(), dim=(-3, -2, -1), norm='ortho')

        # Optional stabilization
        # x_fft = x_fft / (x_fft.abs().mean(dim=(-3,-2,-1), keepdim=True) + 1e-6)

        xr, xi = x_fft.real, x_fft.imag

        # ---- Flatten tokens ----
        B, C3, Dp, Hp, Wp, pd, ph, pwf = xr.shape
        N = Dp * Hp * Wp * pd * ph * pwf

        xr = xr.reshape(B, C3, N)
        xi = xi.reshape(B, C3, N)

        # Real/Imag -> token dimension
        x_tokens = torch.cat([xr, xi], dim=-1)  # [B, 3C, 2N]
        x_tokens = x_tokens.transpose(1, 2)     # [B, 2N, 3C]

        # ---- Split Q K V ----
        q_all, k_all, v_all = torch.split(x_tokens, [self.dim]*3, dim=-1)

        d_head = self.dim // self.num_heads

        def to_heads(t):
            return t.reshape(B, 2*N, self.num_heads, d_head).transpose(1, 2)

        q, k, v = map(to_heads, (q_all, k_all, v_all))

        # ---- Inner TTT update ----
        w1, w2 = self.inner_train_simplified_swiglu(k, v, self.w1, self.w2)

        # ---- SWIGLU ----
        x1 = (q @ w1) * F.silu(q @ w2)  # [B, H, 2N, d]
        x1 = x1.transpose(1, 2).reshape(B, 2*N, self.dim)

        # ---- Back to real/imag ----
        x1 = x1.transpose(1, 2)  # [B, C, 2N]
        xr, xi = torch.chunk(x1, 2, dim=-1)

        xr = xr.reshape(B, self.dim, Dp, Hp, Wp, pd, ph, pwf)
        xi = xi.reshape(B, self.dim, Dp, Hp, Wp, pd, ph, pwf)

        x_fft_new = torch.complex(xr, xi)

        # ---- Inverse FFT ----
        x_patch_back = torch.fft.irfftn(
            x_fft_new,
            s=(p, p, p),
            dim=(-3, -2, -1),
            norm='ortho'
        )

        # ---- Unpatchify ----
        x_out = rearrange(
            x_patch_back,
            'b c d h w pd ph pw -> b c (d pd) (h ph) (w pw)'
        ) 
        x_out = self.proj(x_out)

        return x_out

##########################################################################
class BasicBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(BasicBlock, self).__init__()


        self.norm1 = LayerNorm(dim, LayerNorm_type) 
        self.norm2 = LayerNorm(dim, LayerNorm_type) 
        self.att1 = SpaTTT3D(dim, num_heads, bias) 
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias) 

        self.norm3 = LayerNorm(dim, LayerNorm_type) 
        self.norm4 = LayerNorm(dim, LayerNorm_type) 
        self.att2 = FreqTTT3D(dim, num_heads, bias=bias)
        self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias) 

    def forward(self, x): 

        x = x + self.att1(self.norm1(x))
        x = x + self.ffn1(self.norm2(x))

        x = x + self.att2(self.norm3(x))
        x = x + self.ffn2(self.norm4(x))

        return x 

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks):
        super(TransformerBlock, self).__init__()
        
        self.block_list = nn.Sequential(*[BasicBlock(dim=dim, 
                                                     num_heads=num_heads, 
                                                     ffn_expansion_factor=ffn_expansion_factor, 
                                                     bias=bias, 
                                                     LayerNorm_type=LayerNorm_type, 
                                                     )
                                          for i in range(num_blocks)])
    def forward(self, x): 
        for blk in self.block_list:
            x = blk(x)
        return x

class PixelShuffle3D(nn.Module):

    def __init__(self, scale: int):
        super().__init__()
        if not isinstance(scale, int) or scale < 1:
            raise ValueError("scale must be an integer >= 1")
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 1:
            return x
        if x.dim() != 5:
            raise ValueError("Input must be a 5D tensor (N, C, D, H, W).")
        r = self.scale
        n, c, d, h, w = x.size()
        r3 = r ** 3
        if c % r3 != 0:
            raise ValueError(f"Number of channels ({c}) must be divisible by scale^3 ({r3}).")
        oc = c // r3  # output channels
        # reshape -> permute -> reshape to interleave spatial dims
        x = x.contiguous().view(n, oc, r, r, r, d, h, w)
        # permute to (n, oc, d, r, h, r, w, r)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        # merge interleaved dims -> (n, oc, d*r, h*r, w*r)
        x = x.view(n, oc, d * r, h * r, w * r)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(scale={self.scale})"


class PixelUnshuffle3D(nn.Module):

    def __init__(self, scale: int):
        super().__init__()
        if not isinstance(scale, int) or scale < 1:
            raise ValueError("scale must be an integer >= 1")
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 1:
            return x
        if x.dim() != 5:
            raise ValueError("Input must be a 5D tensor (N, C, D, H, W).")
        r = self.scale
        n, c, d, h, w = x.size()
        if (d % r != 0) or (h % r != 0) or (w % r != 0):
            raise ValueError(f"Depth/Height/Width must be divisible by scale ({r}). "
                             f"Got D={d}, H={h}, W={w}.")
        d_out, h_out, w_out = d // r, h // r, w // r
        # shape to separate r factors
        x = x.contiguous().view(n, c, d_out, r, h_out, r, w_out, r)
        # permute to (n, c, r, r, r, d_out, h_out, w_out)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        # merge r factors into channel dimension -> (n, c * r^3, d_out, h_out, w_out)
        x = x.view(n, c * (r ** 3), d_out, h_out, w_out)
        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat//4, kernel_size=3, stride=1, padding=1, bias=False),
                                  PixelUnshuffle3D(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat*4, kernel_size=3, stride=1, padding=1, bias=False),
                                  PixelShuffle3D(2))

    def forward(self, x):
        return self.body(x) 


class U_TTT(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
        dim = 24,
        num_blocks = [1,2,3,4], ## Each block has a S-TTT block and a F-TTT block
        heads = [1,2,4,8],
        ffn_expansion_factor = 2,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        loss_fun = nn.L1Loss()
    ):

        super(U_TTT, self).__init__() 
        


        self.input_embed = nn.Conv3d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias) 
        
        self.encoder_level1 = TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[0])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2 
        self.encoder_level2 = TransformerBlock(dim=int(dim*2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[1])
        
        self.down2_3 = Downsample(int(dim*2)) ## From Level 2 to Level 3 
        self.encoder_level3 = TransformerBlock(dim=int(dim*4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[2])

        self.down3_4 = Downsample(int(dim*4)) ## From Level 3 to Level 4 
        self.latent = TransformerBlock(dim=int(dim*8), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[3])
        
        self.up4_3 = Upsample(int(dim*8)) ## From Level 4 to Level 3
        self.reduce_chan_decoder_level3 = nn.Conv3d(int(dim*8), int(dim*4), kernel_size=1, bias=bias)
        self.decoder_level3 = TransformerBlock(dim=int(dim*4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[2])


        self.up3_2 = Upsample(int(dim*4)) ## From Level 3 to Level 2
        self.reduce_chan_decoder_level2 = nn.Conv3d(int(dim*4), int(dim*2), kernel_size=1, bias=bias)
        self.decoder_level2 = TransformerBlock(dim=int(dim*2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[1])
        
        self.up2_1 = Upsample(int(dim*2))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_decoder_level1 = nn.Conv3d(int(dim*2), dim, kernel_size=1, bias=bias)
        self.decoder_level1 = TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[0])
        
        self.reduce_refinement = nn.Conv3d(int(dim*2), dim, kernel_size=1, bias=bias)
        self.output = nn.Conv3d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias) 
        self.loss_fun = loss_fun

    def forward(self, inp_img, label_img=None): 

        

    
        inp_enc_level1 = self.input_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1) 

        
        inp_enc_level2 = self.down1_2(out_enc_level1) 
        out_enc_level2 = self.encoder_level2(inp_enc_level2) 


        inp_enc_level3 = self.down2_3(out_enc_level2) 
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 


        inp_enc_level4 = self.down3_4(out_enc_level3)      
        latent = self.latent(inp_enc_level4) 
        
                        
        inp_dec_level3 = self.up4_3(latent) 
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_decoder_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        


        inp_dec_level2 = self.up3_2(out_dec_level3) 
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_decoder_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 



        inp_dec_level1 = self.up2_1(out_dec_level2) 
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1) 
        inp_dec_level1 = self.reduce_chan_decoder_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        
        out_img = torch.cat([out_enc_level1, out_dec_level1], dim=1) 
        out_img = self.reduce_refinement(out_img)
        out_img = self.output(out_img) + inp_img
        if label_img is not None: 
            loss = self.loss_fun(out_img, label_img)
            return loss

        else:
            return out_img








def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 


if __name__ == "__main__":
    import os 
    os.environ['CUDA_VISIBLE_DEVICES']='5' 

    import time 
    from thop import profile, clever_format
    
    '''
    !!!!!!!!
    Caution: Please comment out the code related to reparameterization and retain only the 5x5 convolutional layer in the OmniShift.
    !!!!!!!!
    '''
    
    
    x=torch.ones((1, 1, 64, 64, 64)).type(torch.FloatTensor).cuda() 
    model = U_TTT() 
    model.cuda() 
    
    since = time.time()
    y=model(x)
    print("time", time.time()-since) 
    
    flops, params = profile(model, inputs=(x, ))  
    flops, params = clever_format([flops, params], '%.6f') 
    print('flops',flops)
    print('params', params) 
    print(count_parameters(model)/1e6)
    # print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    # print("Params=", str(params/1e6)+'{}'.format("M"))