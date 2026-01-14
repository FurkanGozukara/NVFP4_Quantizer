#!/usr/bin/env python3
"""Inspect safetensors model structure"""

import sys
from pathlib import Path
from safetensors.torch import load_file
import torch

model_path = sys.argv[1] if len(sys.argv) > 1 else None
if not model_path:
    print("Usage: python inspect_model.py <path_to_model.safetensors>")
    sys.exit(1)

print(f"Loading: {model_path}")
state_dict = load_file(model_path)

print(f"\nTotal keys: {len(state_dict)}")

# Check for scale keys
weight_scale_keys = [k for k in state_dict.keys() if 'weight_scale' in k.lower()]
input_scale_keys = [k for k in state_dict.keys() if 'input_scale' in k.lower()]
scale_keys = [k for k in state_dict.keys() if '_scale' in k.lower()]

print(f"\nKeys with 'weight_scale': {len(weight_scale_keys)}")
print(f"Keys with 'input_scale': {len(input_scale_keys)}")
print(f"Keys with '_scale': {len(scale_keys)}")

# Show first few scale keys
if scale_keys:
    print(f"\nFirst 10 scale keys:")
    for k in scale_keys[:10]:
        tensor = state_dict[k]
        print(f"  {k}")
        print(f"    dtype: {tensor.dtype}, shape: {tensor.shape}")

# Check for weight keys
weight_keys = [k for k in state_dict.keys() if k.endswith('.weight') or k.endswith('weight')]
print(f"\nWeight keys: {len(weight_keys)}")
if weight_keys:
    print(f"\nFirst 5 weight keys and their dtypes:")
    for k in weight_keys[:5]:
        tensor = state_dict[k]
        print(f"  {k}: dtype={tensor.dtype}, shape={tensor.shape}")

# Check for any uint8 tensors
uint8_keys = [k for k in state_dict.keys() if state_dict[k].dtype == torch.uint8]
print(f"\nuint8 tensors: {len(uint8_keys)}")
if uint8_keys:
    print("First 5 uint8 keys:")
    for k in uint8_keys[:5]:
        print(f"  {k}: shape={state_dict[k].shape}")

# Check for float8 tensors
float8_keys = [k for k in state_dict.keys() if 'float8' in str(state_dict[k].dtype)]
print(f"\nfloat8 tensors: {len(float8_keys)}")
if float8_keys:
    print("First 5 float8 keys:")
    for k in float8_keys[:5]:
        print(f"  {k}: dtype={state_dict[k].dtype}, shape={state_dict[k].shape}")

# Show all unique dtypes
dtypes = {}
for k, v in state_dict.items():
    dtype_str = str(v.dtype)
    if dtype_str not in dtypes:
        dtypes[dtype_str] = []
    dtypes[dtype_str].append(k)

print(f"\n=== All dtypes in model ===")
for dtype, keys in sorted(dtypes.items()):
    print(f"{dtype}: {len(keys)} tensors")
    if len(keys) <= 3:
        for k in keys:
            print(f"  - {k}")

# Look for specific NVFP4 patterns
print(f"\n=== Checking for NVFP4 patterns ===")
for pattern in ['weight_scale_2', 'block_scale', 'nvfp4', 'fp4']:
    matching = [k for k in state_dict.keys() if pattern in k.lower()]
    if matching:
        print(f"Found '{pattern}': {len(matching)} keys")
        for k in matching[:3]:
            print(f"  - {k}")
