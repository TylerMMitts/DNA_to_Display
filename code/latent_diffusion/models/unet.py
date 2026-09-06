# The denoising UNet, and the attention blocks it is built from.
#
# This is where the genotype actually reaches the image: cross-attention lets
# each spatial position of the latent read from the SNP embedding tokens.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, t):
        # Creates a vector of sinusoidal embeddings for the given timesteps t
        # Basically creates a version of t that the neural network can actually process
        # Acts as a fingerprint for each specific timestep
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb 


class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, timestep_dim=512):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_ch), num_channels=in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        
        # Timestep projection, maps the timestep embedding to the number of output channels
        self.t_proj = nn.Linear(timestep_dim, out_ch)
        
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
        # Skip connection, adds the input to the output of the block, allowing gradients to flow more easily and helping with training deeper networks
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t_emb):

        # Saves the input tensor
        residual = self.skip(x)
        
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # Add timestep conditioning
        # unsqueeze is used to add two dimensions to the timestep embedding so that it can be broadcasted across the spatial dimensions of the feature map
        # [B, C] -> [B, C, 1, 1]
        h = h + self.t_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        
        # Adds the residual connection to the output of the convolutional layers
        return h + residual


class CrossAttentionBlock(nn.Module):
    
    def __init__(self, channels, snp_embed_dim, d_attention, store_attention=True):
        super().__init__()
        self.d_attention = d_attention
        self.store_attention = store_attention
        
        # Q from UNet, K and V from kinship encoder
        self.W_Q = nn.Linear(channels, d_attention, bias=False)
        self.W_K = nn.Linear(snp_embed_dim, d_attention, bias=False)
        self.W_V = nn.Linear(snp_embed_dim, d_attention, bias=False)
        self.W_out = nn.Linear(d_attention, channels, bias=False)
        
        self.norm = nn.LayerNorm(channels)

        # 10% chance for each attention weight to be set to zero during training, which helps prevent overfitting
        self.dropout = nn.Dropout(0.1)
        
        # Store attention weights for visualization
        self.attention_weights = None
        self.attention_history = []
        
    def forward(self, x, snp_embedding):

        B, C, H, W = x.shape
        N = H * W
        
        # Save residual
        residual = x
        
        # Flatten and normalize
        x_flat = x.view(B, C, N).permute(0, 2, 1)  # [B, N, C]
        x_norm = self.norm(x_flat)
        
        # Compute Q, K, V
        Q = self.W_Q(x_norm) 
        K = self.W_K(snp_embedding) 
        V = self.W_V(snp_embedding) 
        
        # Compute attention scores
        scores = Q @ K.transpose(1, 2) / math.sqrt(self.d_attention)

        # The attention is the weights saying how much each position should listen to each SNP token
        # Softmax converts the scores into probabilities that sum to 1
        attention = F.softmax(scores, dim=-1)

        # Store attention weights for analysis
        if self.store_attention:
            self.attention_weights = attention.detach()
            self.attention_history.append(attention.detach().cpu())

        self.attention_weights = attention.detach()
        attention = self.dropout(attention)
        
        # Apply attention to values
        attended = attention @ V
        
        # Project back to channel dimension
        out = self.W_out(attended)  # [B, N, C]
        
        # Reshape back to spatial
        out = out.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]
        
        # Residual connection
        return out + residual

    def clear_attention_history(self):
        self.attention_history = []
        self.attention_weights = None


class SelfAttentionBlock(nn.Module):
    
    def __init__(self, channels, d_attention=None):
        super().__init__()
        d_attention = d_attention or channels // 2
        self.d_attention = d_attention
        
        self.W_Q = nn.Linear(channels, d_attention, bias=False)
        self.W_K = nn.Linear(channels, d_attention, bias=False)
        self.W_V = nn.Linear(channels, d_attention, bias=False)
        self.W_out = nn.Linear(d_attention, channels, bias=False)
        
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):

        # Self attention allows every pixel to gather context from every other pixel in the feature map
        # While with convolutions, its only in the local window
        # The trade-off is that it is more expensive computationally then convolutions
        B, C, H, W = x.shape
        N = H * W
        
        residual = x
        
        x_flat = x.view(B, C, N).permute(0, 2, 1)  # [B, N, C]
        x_norm = self.norm(x_flat)
        
        Q = self.W_Q(x_norm)
        K = self.W_K(x_norm)
        V = self.W_V(x_norm)
        
        # Compares how relevant each key is to each query 
        scores = Q @ K.transpose(1, 2) / math.sqrt(self.d_attention)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        attended = attention @ V
        out = self.W_out(attended)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        
        return out + residual

class DownBlock(nn.Module):

    def __init__(self, in_ch, out_ch, timestep_dim=512, snp_embed_dim=512,
                 d_attention=512, num_res_blocks=2,
                 has_cross_attn=False, has_self_attn=False, store_attention=True):
        super().__init__()
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResidualBlock(in_ch if i == 0 else out_ch, out_ch, timestep_dim)
            )
        
        # Attention blocks
        self.cross_attn = CrossAttentionBlock(out_ch, snp_embed_dim, d_attention, store_attention=store_attention) if has_cross_attn else None
        self.self_attn = SelfAttentionBlock(out_ch) if has_self_attn else None
        
        # Downsample, reduced the spatial dimensions by half (stride=2)
        # Ex: 32x32 -> 16x16
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x, t_emb, snp_embedding):
        for res_block in self.res_blocks:
            x = res_block(x, t_emb)
        
        if self.cross_attn is not None:
            x = self.cross_attn(x, snp_embedding)
        
        if self.self_attn is not None:
            x = self.self_attn(x)
        
        # Save for skip connection
        skip = x 
        x = self.downsample(x)
        
        return x, skip


class UpBlock(nn.Module):

    def __init__(self, in_ch, out_ch, timestep_dim=512, snp_embed_dim=512,
                 d_attention=512, num_res_blocks=2,
                 has_cross_attn=False, has_self_attn=False, store_attention=True):
        super().__init__()
        
        # Upsample, doubles the spatial dimensions by using nearest neighbor interpolation followed by a convolution
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        
        # Residual blocks (input has extra channels from skip connection)
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResidualBlock(out_ch * 2 if i == 0 else out_ch, out_ch, timestep_dim)
            )
        
        # Attention blocks
        self.cross_attn = CrossAttentionBlock(out_ch, snp_embed_dim, d_attention, store_attention=store_attention) if has_cross_attn else None
        self.self_attn = SelfAttentionBlock(out_ch) if has_self_attn else None
        
    def forward(self, x, skip, t_emb, snp_embedding):
        x = self.upsample(x)
        # Concatenate the skip connection from the down path with the upsampled features
        x = torch.cat([x, skip], dim=1) # [B, C*2, H*2, W*2]
        
        for res_block in self.res_blocks:
            x = res_block(x, t_emb)
        
        if self.cross_attn is not None:
            x = self.cross_attn(x, snp_embedding)
        
        if self.self_attn is not None:
            x = self.self_attn(x)
        
        return x

class DenoisingUNet(nn.Module):
    
    def __init__(self,
                 latent_channels=4,          # From LiteVAE
                 base_channels=128,          # Starting channel count
                 snp_embed_dim=512,          # Kinship embedding dimension
                 d_attention=512,            # Cross-attention dimension
                 num_res_blocks=2,           # Residual blocks per stage
                 attention_resolutions=[1, 2, 4],   # Where to put attention
                 store_attention=True): 
        
        super().__init__()

        self.store_attention = store_attention
        
        # Timestep embedding
        self.t_embed = nn.Sequential(
            # Converts timestep t into 128 dimensional vector
            TimestepEmbedding(128),
            # Projects the vector to 512 dimensions
            nn.Linear(128, 512),
            # Adds non-linearlity
            nn.SiLU(),
            # Another layer for deeper representation (more parameters, more expressive)
            nn.Linear(512, 512),
            nn.SiLU(),
        )
        
        # Projects the 4 channel latent code into [base_channels] channels for the UNet
        self.init_conv = nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1)
        
        # Down path
        self.down_blocks = nn.ModuleList()
        channels = [base_channels, base_channels*2, base_channels*4, base_channels*8]
        
        for i, out_ch in enumerate(channels):
            # Determine if this block should have cross-attention or self-attention based on the resolution
            has_cross_attn = i in attention_resolutions
            has_self_attn = i in attention_resolutions and i > 1
            
            block = DownBlock(
                in_ch=base_channels if i == 0 else channels[i-1],
                out_ch=out_ch,
                timestep_dim=512,
                snp_embed_dim=snp_embed_dim,
                d_attention=d_attention,
                num_res_blocks=num_res_blocks,
                store_attention=store_attention,
                has_cross_attn=has_cross_attn,
                has_self_attn=has_self_attn,
            )
            self.down_blocks.append(block)
        
        # Bottleneck (most compressed layer)
        # This is at the point of the network where information is most abstract
        # so adding parameters here allows the network to learn complex relationships 
        # between the latent representation and the kinship embedding on a global scale
        bottleneck_ch = base_channels * 8
        self.bottleneck = nn.ModuleList([
            ResidualBlock(bottleneck_ch, bottleneck_ch, 512),
            CrossAttentionBlock(bottleneck_ch, snp_embed_dim, d_attention, store_attention=store_attention),
            ResidualBlock(bottleneck_ch, bottleneck_ch, 512),
            SelfAttentionBlock(bottleneck_ch),
            ResidualBlock(bottleneck_ch, bottleneck_ch, 512),
        ])
        
        # Up path
        self.up_blocks = nn.ModuleList()
        
        for i, out_ch in enumerate(reversed(channels)):
            has_cross_attn = (len(channels)-1-i) in attention_resolutions
            has_self_attn = (len(channels)-1-i) in attention_resolutions and (len(channels)-1-i) > 1
            
            block = UpBlock(
                in_ch=bottleneck_ch if i == 0 else channels[len(channels)-i],
                out_ch=out_ch,
                timestep_dim=512,
                snp_embed_dim=snp_embed_dim,
                d_attention=d_attention,
                num_res_blocks=num_res_blocks,
                has_cross_attn=has_cross_attn,
                has_self_attn=has_self_attn,
                store_attention=store_attention,
            )
            self.up_blocks.append(block)
        
        # Normalizes the features before the final convolution to predict noise
        self.final_norm = nn.GroupNorm(
            num_groups=min(32, base_channels), 
            num_channels=base_channels
        )

        # Converts [base_channels] channels back to the original latent channels (4) for the final output
        # This is the final layer that predicts the noise in the latent space
        self.final_conv = nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1)
        
    def forward(self, z_t, t, snp_embedding):

        # Timestep embedding
        t_emb = self.t_embed(t)
        
        skips = []
        
        # Down path
        x = self.init_conv(z_t)
        
        for block in self.down_blocks:
            x, skip = block(x, t_emb, snp_embedding)
            skips.append(skip)
        
        # Bottleneck
        for block in self.bottleneck:
            if isinstance(block, CrossAttentionBlock):
                x = block(x, snp_embedding)
            elif isinstance(block, SelfAttentionBlock):
                x = block(x)
            else:
                x = block(x, t_emb)
        
        # Up path
        for block in self.up_blocks:
            skip = skips.pop()
            x = block(x, skip, t_emb, snp_embedding)
        
        # Final layers
        x = self.final_norm(x)
        x = F.silu(x)
        noise_pred = self.final_conv(x)  # [B, 4, 32, 32]
        
        return noise_pred

    def get_all_attention_weights(self):
        attention_dict = {}
        
        # Collect from down blocks
        for i, block in enumerate(self.down_blocks):
            if block.cross_attn is not None and block.cross_attn.attention_weights is not None:
                attention_dict[f'down_{i}'] = block.cross_attn.attention_weights
        
        # Collect from bottleneck
        for i, block in enumerate(self.bottleneck):
            if isinstance(block, CrossAttentionBlock) and block.attention_weights is not None:
                attention_dict[f'bottleneck_{i}'] = block.attention_weights
        
        # Collect from up blocks
        for i, block in enumerate(self.up_blocks):
            if block.cross_attn is not None and block.cross_attn.attention_weights is not None:
                attention_dict[f'up_{i}'] = block.cross_attn.attention_weights
        
        return attention_dict
    
    def iter_cross_attn_blocks(self):
        # Yields (name, block) for every CrossAttentionBlock in the UNet.
        for i, block in enumerate(self.down_blocks):
            if block.cross_attn is not None:
                yield f'down_{i}', block.cross_attn

        for i, block in enumerate(self.bottleneck):
            if isinstance(block, CrossAttentionBlock):
                yield f'bottleneck_{i}', block

        for i, block in enumerate(self.up_blocks):
            if block.cross_attn is not None:
                yield f'up_{i}', block.cross_attn

    def set_store_attention(self, flag):
        # Propagates the flag to every cross-attention block.
        self.store_attention = flag
        for _, block in self.iter_cross_attn_blocks():
            block.store_attention = flag

    def clear_attention_history(self):

        # Clear from down blocks
        for block in self.down_blocks:
            if block.cross_attn is not None:
                block.cross_attn.clear_attention_history()

        # Clear from bottleneck
        for block in self.bottleneck:
            if isinstance(block, CrossAttentionBlock):
                block.clear_attention_history()

        # Clear from up blocks
        for block in self.up_blocks:
            if block.cross_attn is not None:
                block.cross_attn.clear_attention_history()