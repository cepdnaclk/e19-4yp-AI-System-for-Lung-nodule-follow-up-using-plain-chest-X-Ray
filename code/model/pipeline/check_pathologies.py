"""
Quick script to check torchxrayvision pathologies
"""
import torch
import torchxrayvision as xrv

# Load the model
print("Loading TorchXRayVision model...")
model = xrv.models.ResNet(weights="resnet50-res512-all")

# Check pathologies
print(f"All pathologies: {model.pathologies}")
print(f"Number of pathologies: {len(model.pathologies)}")

# Check if Nodule is available
if 'Nodule' in model.pathologies:
    nodule_idx = model.pathologies.index('Nodule')
    print(f"Nodule index: {nodule_idx}")
else:
    print("Nodule not found in pathologies!")
    # Check for similar terms
    nodule_like = [p for p in model.pathologies if 'nodule' in p.lower() or 'mass' in p.lower() or 'lesion' in p.lower()]
    print(f"Similar pathologies: {nodule_like}")

# Check model structure
print(f"Model has {len(list(model.parameters()))} parameter tensors")
print(f"Final layer shape: {list(model.parameters())[-1].shape}")
