# Haar wavelet transform, used by the encoder to split an image into
# frequency sub-bands before any convolution happens.

import torch
import numpy as np
import pywt

# Implements discrete wavelet transform (DWT) for 2D images using Haar wavelets
# Haar wavelet applies 4 different convolution kernels to the input image to produce 4 sub-bands: LL, LH, HL, HH
# The high-frequency sub-bands captures the details of the image, while the low-frequency sub-band captures the overall structure of the image.

def dwt_2d(x):

    # batch_size, channels, height, width = x.shape
    B, C, H, W = x.shape
    
    # Convert to numpy for pywt processing
    x_np = x.cpu().numpy()
    
    LL_list, LH_list, HL_list, HH_list = [], [], [], []
    
    # Perform DWT for each channel of each image in the batch
    for b in range(B):
        LL_b, LH_b, HL_b, HH_b = [], [], [], []
        
        for c in range(C):
            channel = x_np[b, c]  # [H, W]
            coeffs = pywt.dwt2(channel, 'haar')
            LL, (LH, HL, HH) = coeffs
            
            LL_b.append(LL)
            LH_b.append(LH)
            HL_b.append(HL)
            HH_b.append(HH)
        
        LL_list.append(np.stack(LL_b, axis=0))
        LH_list.append(np.stack(LH_b, axis=0))
        HL_list.append(np.stack(HL_b, axis=0))
        HH_list.append(np.stack(HH_b, axis=0))
    
    # Convert back to tensors
    dtype = x.dtype
    device = x.device
    
    LL = torch.tensor(np.stack(LL_list, axis=0), device=device, dtype=dtype)
    LH = torch.tensor(np.stack(LH_list, axis=0), device=device, dtype=dtype)
    HL = torch.tensor(np.stack(HL_list, axis=0), device=device, dtype=dtype)
    HH = torch.tensor(np.stack(HH_list, axis=0), device=device, dtype=dtype)
    
    return LL, LH, HL, HH