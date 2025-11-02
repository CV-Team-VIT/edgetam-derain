# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import List, Dict, Tuple

# --- Model Components ---
from image_encoder import ImageEncoder, FpnNeck
from timm import TimmBackbone
from position_encoding import PositionEmbeddingSine
from memory_attention import MemoryAttention, MemoryAttentionLayer
# Assuming sam2 path is correct for relative import
from sam2.modeling.sam.transformer import RoPEAttention 
from image_memory_encoder import MemoryEncoder, ImageDownSampler, CXBlock, Fuser
from unet_decoder import UNetDecoder


class DerainTAM(nn.Module):
    """
    Implements DerainTAM, a recurrent video deraining model.
    
    This model leverages a lightweight encoder, temporal memory attention,
    and a reconstructive U-Net decoder to process videos frame-by-frame.
    """
    def __init__(
        self,
        image_encoder: ImageEncoder,
        memory_attention: MemoryAttention,
        memory_encoder: MemoryEncoder,
        unet_decoder: UNetDecoder,
        num_maskmem: int = 7, # Number of memory frames to keep
    ):
        super().__init__()
        
        self.image_encoder = image_encoder
        self.memory_attention = memory_attention
        self.memory_encoder = memory_encoder
        self.unet_decoder = unet_decoder
        
        self.num_maskmem = num_maskmem
        self.hidden_dim = memory_attention.d_model
        
        # Fallback token for the first frame when no memory exists
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        
        # Learnable temporal positional encoding for the memory bank
        self.maskmem_tpos_enc = nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.hidden_dim)
        )

    def forward_frame(
        self, 
        rainy_frame: torch.Tensor, 
        memory_bank: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Processes a single rainy frame using temporal memory.
        
        Args:
            rainy_frame: The current rainy frame [B, 3, H, W].
            memory_bank: A list of memory dicts from previous frames.
                                      
        Returns:
            A tuple containing:
            - derained_image: The predicted clean frame [B, 3, H, W].
            - new_memory: The memory computed from this frame's output.
        """
        
        B = rainy_frame.shape[0]
        device = rainy_frame.device
        
        # --- 1. ENCODE ---
        # Extract multi-scale features from the current rainy frame
        encoder_out = self.image_encoder(rainy_frame)
        pix_feat = encoder_out["vision_features"] # Low-res features for attention
        fpn_skip_features = encoder_out["backbone_fpn"] # Multi-scale features for UNet skips
        
        # Prepare features for Transformer (seq-first format: [HW, B, C])
        current_vision_feats = [pix_feat.flatten(2).permute(2, 0, 1)]
        current_vision_pos = [encoder_out["vision_pos_enc"][-1].flatten(2).permute(2, 0, 1)]
        
        
        # --- 2. PREPARE & FUSE MEMORY ---
        to_cat_memory = []
        to_cat_memory_pos_embed = []

        if not memory_bank:
            # Use the "no_mem" token if memory bank is empty (e.g., first frame)
            to_cat_memory.append(self.no_mem_embed.expand(1, B, self.hidden_dim))
            to_cat_memory_pos_embed.append(self.no_mem_pos_enc.expand(1, B, self.hidden_dim))
        else:
            # Populate memory and add temporal positional encodings
            for t_pos, mem in enumerate(memory_bank):
                mem_feat = mem["vision_features"].to(device).flatten(2).permute(2, 0, 1)
                mem_pos = mem["vision_pos_enc"][0].to(device).flatten(2).permute(2, 0, 1)
                
                to_cat_memory.append(mem_feat)
                # Add learned temporal encoding
                mem_pos_with_temporal = mem_pos + self.maskmem_tpos_enc[t_pos]
                to_cat_memory_pos_embed.append(mem_pos_with_temporal)

        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        # Fuse current frame features with past memories via attention
        fused_features_flat = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=0, # Not used for this task
            num_spatial_mem=len(to_cat_memory),
        )
        
        # Reshape fused features back to 2D [B, C, H_low, W_low]
        B, C, H_low, W_low = pix_feat.shape
        fused_features = fused_features_flat.permute(1, 2, 0).view(B, C, H_low, W_low)
        
        
        # --- 3. DECODE (DERAIN) ---
        # Reconstruct the clean image using the U-Net decoder
        derained_image = self.unet_decoder(
            fused_features=fused_features,
            fpn_skip_features=fpn_skip_features
        )
        
        # --- 4. CREATE NEW MEMORY ---
        # Encode the *derained* image and *original* features into a new memory
        new_memory = self.memory_encoder(
            pix_feat=pix_feat, # Features from the *rainy* frame
            derained_image=derained_image # Output *clean* frame
        )
        
        # --- 5. RETURN ---
        return derained_image, new_memory


# --- This is how you would build the model ---
def build_derain_tam():
    """
    Helper function to build and instantiate the full DerainTAM model.
    Note: Channel dimensions and layer counts are examples and must be
    verified to match the pre-trained weights or model definitions.
    """
    
    # --- 1. Build Image Encoder (RepViT-M1) ---
    d_model = 256 # Example embedding dimension
    # Example channels: [high-res, mid-res, low-res]. Must match the timm model.
    backbone_channels = [64, 128, 256] 
    
    timm_backbone = TimmBackbone(
        name='repvit_m1_dist', # Ensure this timm model is available
        features=('layer1', 'layer2', 'layer3'), # Example feature layer names
    )
    # TODO: Verify timm_backbone.channel_list matches backbone_channels
    # backbone_channels = timm_backbone.channel_list[::-1] # [high, mid, low]
    
    pos_encoding_module = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)
    
    fpn_neck = FpnNeck(
        position_encoding=pos_encoding_module,
        d_model=d_model,
        backbone_channel_list=backbone_channels[::-1] # FpnNeck expects [low, mid, high]
    )
    
    image_encoder = ImageEncoder(trunk=timm_backbone, neck=fpn_neck)

    # --- 2. Build Memory Attention ---
    mem_attn_layer = MemoryAttentionLayer(
        activation="relu",
        cross_attention=RoPEAttention(d_model, 8), # d_model, num_heads
        d_model=d_model,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=True,
        self_attention=RoPEAttention(d_model, 8), # d_model, num_heads
    )
    memory_attention = MemoryAttention(
        d_model=d_model,
        pos_enc_at_input=True,
        layer=mem_attn_layer,
        num_layers=2 # Example number of layers
    )
    
    # --- 3. Build Memory Encoder ---
    img_downsampler = ImageDownSampler(embed_dim=d_model, total_stride=16) # Must match encoder stride
    cx_block = CXBlock(dim=d_model)
    fuser = Fuser(layer=cx_block, num_layers=4) # Example number of layers
    
    memory_encoder = MemoryEncoder(
        out_dim=d_model,
        image_downsampler=img_downsampler,
        fuser=fuser,
        position_encoding=pos_encoding_module,
        in_dim=d_model
    )
    
    # --- 4. Build UNet Decoder ---
    unet_decoder = UNetDecoder(
        fpn_channels=backbone_channels, # [high, mid, low]
        decoder_channels=[256, 128, 64] # Example decoder channels
    )

    # --- 5. Build Final Model ---
    model = DerainTAM(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        unet_decoder=unet_decoder
    )
    
    return model

if __name__ == '__main__':
    # This is a test script to validate model instantiation and a single forward pass.
    
    try:
        model = build_derain_tam()
        model.eval()
        
        # Test parameters
        B = 1 # Batch size
        T = 5 # Number of frames
        H, W = 256, 256 # Image size
        
        video_clip = torch.randn(T, B, 3, H, W)
        memory_bank = [] # Initialize empty memory bank
        
        print("--- Building DerainTAM model ---")
        # Calculate and print total trainable parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {params/1e6:.2f}M")
        
        print("\n--- Running recurrent frame test ---")
        with torch.no_grad():
            for t in range(T):
                rainy_frame = video_clip[t]
                
                # Process one frame
                derained_image, new_memory = model.forward_frame(rainy_frame, memory_bank)
                
                # Update memory bank (FIFO queue)
                if len(memory_bank) >= (model.num_maskmem - 1):
                    memory_bank.pop(0) # Remove oldest memory
                memory_bank.append(new_memory)
                
                print(f"Frame {t}:")
                print(f"  > Derained image shape: {derained_image.shape}")
                print(f"  > New memory features shape: {new_memory['vision_features'].shape}")
        
        print("\nModel build and forward pass successful!")

    except ImportError as e:
        print(f"ImportError: {e}")
        print("Please ensure all dependencies (timm, sam2) are installed and in your PYTHONPATH.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("This is likely a channel mismatch in the `build_derain_tam` function.")
        print("Please verify all `d_model` and `backbone_channels` are correct.")

