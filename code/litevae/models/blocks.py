import torch
import torch.nn as nn

# Creates class that inherits from nn.Module for the LiteVAE encoder and decoder blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        # GroupNorm normalizes across groups of channels rather than across the batch dimension
        # Use min(32, num_channels) to ensure num_groups doesn't exceed num_channels
        # This is more stable than BatchNorm when batch size is small
        num_groups_in = min(32, in_ch)
        self.norm1 = nn.GroupNorm(num_groups=num_groups_in, num_channels=in_ch)
        
        # Activation adds non-linearity to the model, allowing it to learn more complex functions
        self.act1 = nn.SiLU()
        
        # Convolution layer that converts an input tensor that has in_ch channels to an output tensor that has out_ch channels
        # Parameters the model will learn: in_ch * out_ch * kernel_size^2 + out_ch (for bias)
        # kernel_size=3 and padding=1 means the output will have the same spatial dimensions as the input
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # Use min(32, num_channels) to ensure num_groups doesn't exceed num_channels
        num_groups_out = min(32, out_ch)
        self.norm2 = nn.GroupNorm(num_groups=num_groups_out, num_channels=out_ch)
        
        # Activation adds non-linearity to the model, allowing it to learn more complex functions
        self.act2 = nn.SiLU()
        
        # Parameters of the second convolution layer will learn more complex features from the output of the first convolution layer
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # If the number of input channels is not equal to the number of output channels, we need to use a 1x1 convolution to match the dimensions for the residual connection. Otherwise, we can use an identity mapping (no change).
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        # Saves the input tensor as a residual connection, which will be added back to the output of the convolutional layers
        residual = self.skip(x)

        # Applies GroupNorm to normalize the input across channel groups
        h = self.norm1(x)
        
        # Applies the activation function
        h = self.act1(h)
        
        # Applies the first convolution layer
        h = self.conv1(h)

        # Applies GroupNorm to normalize the output across channel groups
        h = self.norm2(h)
        
        # Applies the activation function
        h = self.act2(h)
        
        # Applies the second convolution layer
        h = self.conv2(h)

        # Adds the residual connection to the output of the convolutional layers
        # This allows the model to add the positive learned feature to the original input and subtract the negative learned feature from the original input
        return h + residual


class SelfModulatedConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        
        # Learnable scale parameter for each input channel
        # This controls how much each input channel contributes to the output, allowing the model to learn which features are the most important
        self.scale = nn.Parameter(torch.ones(in_ch))
        
        # Standard convolution
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        
        # Creates a learnable bias parameter for each output channel
        # It allows the model to shift the output of the convolution, which can help with learning more complex features
        self.bias = nn.Parameter(torch.zeros(out_ch))
        
        # Activation after modulation
        self.act = nn.SiLU()
    
    def forward(self, x):
        # Apply modulation to convolution weights
        # w' = (s_i * w) / sqrt(sum(s_i * w)^2 + epsilon)
        weights = self.conv.weight

        # Reshapes scale to match the dimensions of the convolution weights for broadcasting
        # The scale parameter is reshaped to have dimensions (1, in_ch, 1, 1) so that it can be multiplied with the convolution weights
        scale = self.scale.view(1, -1, 1, 1)  
        
        # Modulate weights by scaling each input channel
        modulated_weights = scale * weights
        
        # Computes the L2 norm of the modulated weights for normalization
        # Prevents the weights from growing too large, which can destabilize training
        norm = torch.sqrt((modulated_weights ** 2).sum(dim=(1, 2, 3), keepdim=True) + 1e-8)
        normalized_weights = modulated_weights / norm
        
        # Apply convolution with modulated weights
        out = torch.nn.functional.conv2d(x, normalized_weights, bias=None, padding=self.conv.padding)
        
        # Add bias
        out = out + self.bias.view(1, -1, 1, 1)
        
        # Apply activation
        return self.act(out)


class SMCResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        # SMC modulates the convolution weights to balance feature magnitudes
        # Reduces channels and extracts features
        self.smc1 = SelfModulatedConv2d(in_ch, out_ch, kernel_size=3, padding=1)
        # Refines features and keeps the same number of channels
        self.smc2 = SelfModulatedConv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
        # Skip connection to preserve the input information
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        # Saves the input tensor as a residual connection
        residual = self.skip(x)
        
        # Applies self-modulated convolutions
        x = self.smc1(x)
        x = self.smc2(x)
        
        # Adds the residual connection to the output
        return x + residual


class LiteVAEUNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_res_blocks=3):
        super().__init__()

        # If the number of input channels is not equal to the number of output channels, we need to use a 1x1 convolution to match the dimensions. Otherwise, we can use an identity mapping (no change).
        self.in_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1) if in_ch != out_ch else nn.Identity()

        # Creates [num_res_blocks] residual blocks that will be applied after the initial convolution
        layers = []
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        self.body = nn.Sequential(*layers)

        # Final convolution layer that processes the features after the residual blocks
        self.out_conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        # Applies the initial convolution to match channel dimensions
        x = self.in_conv(x)

        # Applies the series of residual blocks to process features at a fixed resolution
        x = self.body(x)

        # Applies the final convolution to refine the features
        x = self.out_conv(x)
        
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_res_blocks=2):
        super().__init__()

        # Creates a sequential container that chains multiple operations together   
        self.upsample = nn.Sequential(
            # Doubles the spatial dimensions of the input tensor using nearest neighbor interpolation
            nn.Upsample(scale_factor=2, mode='nearest'),
            # In the upsampling process, this reduces the number of channels from in_ch to out_ch and smoothes the image from the upsampling operation
            SelfModulatedConv2d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        
        # Creates [num_res_blocks] residual blocks that will be applied after the upsampling operation
        # Each residual block uses GroupNorm for stable training with small batch sizes
        # This allows the model to learn more complex features after the upsampling operation, which is important for reconstructing high-quality images from the latent code
        self.res_blocks = nn.Sequential(*[SMCResidualBlock(out_ch, out_ch) for _ in range(num_res_blocks)])
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.res_blocks(x)
        return x