# APTM_Retrieval(
  ## (vision_encoder): SwinTransformer(
    (patch_embed): PatchEmbed(
      (proj): Conv2d(3, 128, kernel_size=(4, 4), stride=(4, 4))
      (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    )
    (pos_drop): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0): BasicLayer(
        dim=128, input_resolution=(96, 32), depth=2
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=128, input_resolution=(96, 32), num_heads=4, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=128, window_size=(8, 8), num_heads=4
              (qkv): Linear(in_features=128, out_features=384, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=128, out_features=128, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=512, out_features=128, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=128, input_resolution=(96, 32), num_heads=4, window_size=8, shift_size=4, mlp_ratio=4.0
            (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=128, window_size=(8, 8), num_heads=4
              (qkv): Linear(in_features=128, out_features=384, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=128, out_features=128, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=512, out_features=128, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          input_resolution=(96, 32), dim=128
          (reduction): Linear(in_features=512, out_features=256, bias=False)
          (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
      )
      (1): BasicLayer(
        dim=256, input_resolution=(48, 16), depth=2
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=256, input_resolution=(48, 16), num_heads=8, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=256, window_size=(8, 8), num_heads=8
              (qkv): Linear(in_features=256, out_features=768, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=256, out_features=256, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=256, out_features=1024, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=1024, out_features=256, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=256, input_resolution=(48, 16), num_heads=8, window_size=8, shift_size=4, mlp_ratio=4.0
            (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=256, window_size=(8, 8), num_heads=8
              (qkv): Linear(in_features=256, out_features=768, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=256, out_features=256, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=256, out_features=1024, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=1024, out_features=256, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          input_resolution=(48, 16), dim=256
          (reduction): Linear(in_features=1024, out_features=512, bias=False)
          (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
      (2): BasicLayer(
        dim=512, input_resolution=(24, 8), depth=18
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (2): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (3): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (4): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (5): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (6): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (7): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (8): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (9): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (10): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (11): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (12): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (13): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (14): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (15): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (16): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (17): SwinTransformerBlock(
            dim=512, input_resolution=(24, 8), num_heads=16, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(8, 8), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          input_resolution=(24, 8), dim=512
          (reduction): Linear(in_features=2048, out_features=1024, bias=False)
          (norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        )
      )
      (3): BasicLayer(
        dim=1024, input_resolution=(12, 4), depth=2
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=1024, input_resolution=(12, 4), num_heads=32, window_size=4, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=1024, window_size=(4, 4), num_heads=32
              (qkv): Linear(in_features=1024, out_features=3072, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=1024, out_features=1024, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=1024, input_resolution=(12, 4), num_heads=32, window_size=4, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=1024, window_size=(4, 4), num_heads=32
              (qkv): Linear(in_features=1024, out_features=3072, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=1024, out_features=1024, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath()
            (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (act): GELU()
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
      )
    )
    (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    (avgpool): AdaptiveAvgPool1d(output_size=1)
  )
  ## (text_encoder): BertForMaskedLM(
    (bert): BertModel(
      (embeddings): BertEmbeddings(
        (word_embeddings): Embedding(30522, 768, padding_idx=0)
        (position_embeddings): Embedding(512, 768)
        (token_type_embeddings): Embedding(2, 768)
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (encoder): BertEncoder(
        (layer): ModuleList(
          (0): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (1): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (2): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (3): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (4): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (5): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (6): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (crossattention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=1024, out_features=768, bias=True)
                (value): Linear(in_features=1024, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (7): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (crossattention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=1024, out_features=768, bias=True)
                (value): Linear(in_features=1024, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (8): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (crossattention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=1024, out_features=768, bias=True)
                (value): Linear(in_features=1024, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (9): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (crossattention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=1024, out_features=768, bias=True)
                (value): Linear(in_features=1024, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (10): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (crossattention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=1024, out_features=768, bias=True)
                (value): Linear(in_features=1024, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (11): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (crossattention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=1024, out_features=768, bias=True)
                (value): Linear(in_features=1024, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
    )
    (cls): BertOnlyMLMHead(
      (predictions): BertLMPredictionHead(
        (transform): BertPredictionHeadTransform(
          (dense): Linear(in_features=768, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
        (decoder): Linear(in_features=768, out_features=30522, bias=True)
      )
    )
  )
  ## (vision_proj): Linear(in_features=1024, out_features=256, bias=True)
  ## (text_proj): Linear(in_features=768, out_features=256, bias=True)
  ## (itm_head): Sequential(
    (0): Linear(in_features=768, out_features=1536, bias=True)
    (1): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
    (2): GELU()
    (3): Linear(in_features=1536, out_features=2, bias=True)
  )
)