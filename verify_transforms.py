import torch
import numpy as np
from PIL import Image
from src.datasets.utils import normed_tensors

def verify_transforms():
    print("=== Verifying Transforms (Resize Fix) ===")
    
    # Simulate a problematic image (e.g., 227x227 from CaffeNet data)
    # create a random PIL image (Height, Width, Channels)
    # Note: PIL uses (Width, Height) for size, but numpy uses (H, W, C)
    arr = np.random.randint(0, 255, (227, 227, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    print(f"Input Image Size: {img.size}")
    
    # Apply the transform
    transform = normed_tensors()
    out = transform(img)
    
    print(f"Output Tensor Shape: {out.shape}")
    
    # Assertions
    # expected shape is [C, H, W] = [3, 224, 224]
    assert out.shape == (3, 224, 224), f"Expected (3, 224, 224), got {out.shape}"
    print("✅ SUCCESS: Transform resized correctly.")

if __name__ == "__main__":
    verify_transforms()



