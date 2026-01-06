#!/usr/bin/env python3
"""
Fix checkpoint compatibility by re-saving with NumPy 1.x
This resolves: ModuleNotFoundError: No module named 'numpy._core'
"""

import torch
import sys

def fix_checkpoint(input_path: str, output_path: str = None):
    """
    Load a checkpoint saved with NumPy 2.x and re-save it with NumPy 1.x
    
    Args:
        input_path: Path to the checkpoint file to fix
        output_path: Path to save the fixed checkpoint (defaults to input_path with .fixed suffix)
    """
    if output_path is None:
        output_path = input_path.replace('.pt', '_fixed.pt')
    
    print(f"Loading checkpoint from: {input_path}")
    try:
        # Load with weights_only=False to handle numpy._core issue
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
        print(f"✓ Checkpoint loaded successfully")
        
        # Display checkpoint contents
        if isinstance(checkpoint, dict):
            print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
        
        print(f"\nSaving fixed checkpoint to: {output_path}")
        torch.save(checkpoint, output_path)
        print(f"✓ Fixed checkpoint saved successfully")
        
        # Verify the fixed checkpoint can be loaded
        print(f"\nVerifying fixed checkpoint...")
        _ = torch.load(output_path, map_location='cpu', weights_only=False)
        print(f"✓ Verification successful")
        
        return output_path
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_checkpoint.py <checkpoint_path> [output_path]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    fix_checkpoint(input_path, output_path)
