"""
Quick test to check attention module behavior.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from lightweight_model import LightweightAttentionModule, DebugAttentionModule

def test_attention_modules():
    """Test both attention modules to see their behavior."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test input (simulating DRR)
    batch_size = 1
    height, width = 512, 512
    
    # Create a test DRR with some structure (not just noise)
    test_drr = torch.zeros(batch_size, 1, height, width, device=device)
    
    # Add some synthetic nodule-like structures
    # Central bright region
    center_h, center_w = height // 2, width // 2
    test_drr[:, :, center_h-50:center_h+50, center_w-30:center_w+30] = 0.8
    
    # Side region
    test_drr[:, :, center_h-20:center_h+20, center_w+100:center_w+140] = 0.6
    
    # Add some noise
    noise = torch.randn_like(test_drr) * 0.1
    test_drr = test_drr + noise
    test_drr = torch.clamp(test_drr, 0, 1)
    
    print("Testing Attention Modules...")
    print(f"Input shape: {test_drr.shape}")
    print(f"Input range: [{test_drr.min().item():.3f}, {test_drr.max().item():.3f}]")
    
    # Test original attention module
    original_attention = LightweightAttentionModule(in_channels=1, hidden_channels=16).to(device)
    
    with torch.no_grad():
        original_attention.eval()
        orig_output = original_attention(test_drr)
        print(f"\nOriginal Attention:")
        print(f"  Output shape: {orig_output.shape}")
        print(f"  Output range: [{orig_output.min().item():.6f}, {orig_output.max().item():.6f}]")
        print(f"  Output mean: {orig_output.mean().item():.6f}")
        print(f"  Output std: {orig_output.std().item():.6f}")
        
        # Check if attention is saturated
        unique_values = torch.unique(orig_output).numel()
        print(f"  Unique values: {unique_values}")
        
    # Test debug attention module
    debug_attention = DebugAttentionModule(in_channels=1, hidden_channels=8).to(device)
    
    with torch.no_grad():
        debug_attention.eval()
        debug_output = debug_attention(test_drr)
        print(f"\nDebug Attention:")
        print(f"  Output shape: {debug_output.shape}")
        print(f"  Output range: [{debug_output.min().item():.6f}, {debug_output.max().item():.6f}]")
        print(f"  Output mean: {debug_output.mean().item():.6f}")
        print(f"  Output std: {debug_output.std().item():.6f}")
        
        # Check if attention is saturated
        unique_values = torch.unique(debug_output).numel()
        print(f"  Unique values: {unique_values}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input DRR
    axes[0].imshow(test_drr.cpu().numpy()[0, 0], cmap='gray')
    axes[0].set_title('Input DRR (Synthetic)')
    axes[0].axis('off')
    
    # Original attention
    axes[1].imshow(orig_output.cpu().numpy()[0, 0], cmap='hot')
    axes[1].set_title(f'Original Attention\n(range: {orig_output.min().item():.3f}-{orig_output.max().item():.3f})')
    axes[1].axis('off')
    
    # Debug attention
    axes[2].imshow(debug_output.cpu().numpy()[0, 0], cmap='hot')
    axes[2].set_title(f'Debug Attention\n(range: {debug_output.min().item():.3f}-{debug_output.max().item():.3f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('attention_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved as 'attention_test_comparison.png'")
    
    # Test with real noise to see behavior
    print(f"\n" + "="*50)
    print("Testing with pure noise (worst case):")
    
    noise_input = torch.randn(batch_size, 1, height, width, device=device)
    
    with torch.no_grad():
        orig_noise_output = original_attention(noise_input)
        debug_noise_output = debug_attention(noise_input)
        
        print(f"Original with noise - range: [{orig_noise_output.min().item():.6f}, {orig_noise_output.max().item():.6f}]")
        print(f"Debug with noise - range: [{debug_noise_output.min().item():.6f}, {debug_noise_output.max().item():.6f}]")
    
    return orig_output, debug_output

if __name__ == "__main__":
    test_attention_modules()
