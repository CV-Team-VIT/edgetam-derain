# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class DecoderBlock(nn.Module):
    """
    A single block in the U-Net decoder.
    This block upsamples the input feature map, concatenates it with a skip
    connection from the corresponding encoder layer, and passes it
    through two convolution layers.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        # Upsampling layer: doubles the feature map size
        # in_channels is from the previous, deeper decoder block
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # Convolutional block
        # The input channels will be out_channels (from upsampling) + skip_channels (from encoder)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):
        # 1. Upsample the input from the previous (deeper) layer
        x = self.up(x)
        
        # 2. Concatenate with the skip connection from the encoder
        # Ensure spatial dimensions match if necessary (though FPN should handle this)
        # We might need to handle slight size mismatches due to conv padding
        if x.shape != skip_connection.shape:
            # F.interpolate is more flexible than padding
             x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip_connection], dim=1)
        
        # 3. Pass through the convolutional block
        return self.conv(x)

class UNetDecoder(nn.Module):
    """
    The main U-Net Decoder.
    This module takes the list of feature maps from the ImageEncoder's FPN
    (for skip connections) and the fused feature map from the MemoryAttention
    (as the starting point for upsampling).
    """
    def __init__(self, fpn_channels: List[int], decoder_channels: List[int] = [256, 128, 64]):
        """
        Initializes the UNetDecoder.
        
        Args:
            fpn_channels (List[int]): List of channel numbers for each FPN feature map,
                                      from highest-res to lowest-res.
                                      e.g., [256, 256, 256] from your FpnNeck.
            decoder_channels (List[int]): List of output channels for each decoder block.
                                          The length should match fpn_channels.
        """
        super().__init__()
        
        # The FPN features are typically [high-res, mid-res, low-res]
        # We reverse them to [low-res, mid-res, high-res] to match the decoder's
        # upsampling path.
        self.fpn_channels = fpn_channels[::-1] # Now [low-res, mid-res, high-res]
        
        # The first decoder block takes the fused feature from MemoryAttention
        # The number of channels must match the output of MemoryAttention (e.g., 256)
        # Let's assume the fused feature has the same channel dim as the lowest-res FPN feature.
        in_channels = self.fpn_channels[0]
        
        self.decoder_blocks = nn.ModuleList()
        
        # Create decoder blocks
        # Block 0: Takes fused_feature (low-res) + skip_connection (mid-res)
        # Block 1: Takes output_of_block_0 + skip_connection (high-res)
        # ...and so on
        
        for i in range(len(decoder_channels)):
            skip_ch = self.fpn_channels[i+1] if (i+1) < len(self.fpn_channels) else 0
            out_channels = decoder_channels[i]
            
            self.decoder_blocks.append(
                DecoderBlock(in_channels, skip_ch, out_channels)
            )
            in_channels = out_channels # The output of this block is the input to the next

        # Final output layer to produce a 3-channel (RGB) image
        # It takes the output of the last (highest-res) decoder block
        self.final_conv = nn.Conv2d(in_channels, 3, kernel_size=1)
        
        # A final activation to map the output to a normalized range
        # Tanh is good for [-1, 1], Sigmoid is good for [0, 1]
        self.final_activation = nn.Sigmoid() # Or nn.Tanh()

    def forward(self, fused_features: torch.Tensor, fpn_skip_features: List[torch.Tensor]):
        """
        Args:
            fused_features (torch.Tensor): The output from the MemoryAttention block.
                                           This is the starting point for the decoder.
                                           Shape: [B, C, H_low, W_low]
            fpn_skip_features (List[torch.Tensor]): The list of feature maps from the
                                                    ImageEncoder (backbone_fpn).
                                                    Assumed to be [high-res, mid-res, low-res].
        """
        
        # Reverse the skip features to match the decoder's upsampling path
        # [high-res, mid-res, low-res] -> [low-res, mid-res, high-res]
        skip_features = fpn_skip_features[::-1]
        
        x = fused_features
        
        # Iterate through the decoder blocks
        # On iteration i:
        #   x = output of block i-1
        #   skip = fpn_feature[i+1]
        for i, block in enumerate(self.decoder_blocks):
            # Get the corresponding skip connection (the next one up in resolution)
            skip = skip_features[i+1] if (i+1) < len(skip_features) else None
            
            # If we run out of skip connections, we just upsample without skip
            if skip is None:
                # This part might need a simpler block that just upsamples
                # For now, we'll assume the number of blocks matches the skips
                continue 
                
            x = block(x, skip)

        # 1. Final convolution to get to 3 channels
        x = self.final_conv(x)
        
        # 2. Upsample to the final full image size if needed
        # The last block's output might not be full resolution.
        # We'll upsample it to match the *first* (highest-res) skip feature size.
        orig_size = fpn_skip_features[0].shape[2:] # H_high, W_high
        if x.shape[2:] != orig_size:
             x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        
        # 3. Apply final activation to get the derained image
        derained_image = self.final_activation(x)
        
        return derained_image
