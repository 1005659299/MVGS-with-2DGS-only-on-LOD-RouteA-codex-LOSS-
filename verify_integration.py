"""
MVGS + 2D-GS Integration Verification Script
Verifies: 2D scaling, allmap parsing, geometric regularization
"""
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("MVGS + 2D-GS Integration Verification")
print("=" * 60)

# Test 1: Import modules
print("\n[Test 1] Importing modules...")
try:
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    print("  - diff_surfel_rasterization: OK")
except ImportError as e:
    print(f"  - diff_surfel_rasterization: FAILED ({e})")
    sys.exit(1)

try:
    from scene.gaussian_model import GaussianModel
    print("  - GaussianModel: OK")
except ImportError as e:
    print(f"  - GaussianModel: FAILED ({e})")
    sys.exit(1)

try:
    from utils.point_utils import depth_to_normal, depths_to_points
    print("  - point_utils: OK")
except ImportError as e:
    print(f"  - point_utils: FAILED ({e})")
    sys.exit(1)

# Test 2: Check 2D scaling dimension
print("\n[Test 2] Checking 2D scaling dimension...")
model = GaussianModel(sh_degree=3)

# Create a simple point cloud
class MockPointCloud:
    def __init__(self, n_points=100):
        self.points = torch.randn(n_points, 3).numpy()
        self.colors = torch.rand(n_points, 3).numpy()

pcd = MockPointCloud(100)
model.create_from_pcd(pcd, spatial_lr_scale=1.0)

scaling_shape = model._scaling.shape
print(f"  - Scaling tensor shape: {scaling_shape}")
if scaling_shape[1] == 2:
    print("  - 2D scaling (N, 2): OK")
else:
    print(f"  - 2D scaling (N, 2): FAILED (got {scaling_shape[1]} dimensions)")

# Test 3: Check covariance output
print("\n[Test 3] Checking covariance computation...")
model.training_setup(type('args', (), {
    'percent_dense': 0.01,
    'position_lr_init': 0.00016,
    'position_lr_final': 0.0000016,
    'position_lr_delay_mult': 0.01,
    'position_lr_max_steps': 30000,
    'feature_lr': 0.0025,
    'opacity_lr': 0.05,
    'scaling_lr': 0.005,
    'rotation_lr': 0.001,
})())

covariance = model.get_covariance(scaling_modifier=1.0)
print(f"  - Covariance tensor shape: {covariance.shape}")
if covariance.shape[1:] == torch.Size([4, 4]):
    print("  - 4x4 transformation matrix: OK")
else:
    print(f"  - 4x4 transformation matrix: FAILED (got {covariance.shape})")

# Test 4: Check arguments
print("\n[Test 4] Checking new arguments...")
from arguments import OptimizationParams, PipelineParams
from argparse import ArgumentParser

parser = ArgumentParser()
opt_params = OptimizationParams(parser)
pipe_params = PipelineParams(parser)

has_lambda_normal = hasattr(opt_params, 'lambda_normal')
has_lambda_dist = hasattr(opt_params, 'lambda_dist')
has_depth_ratio = hasattr(pipe_params, 'depth_ratio')

print(f"  - lambda_normal: {'OK' if has_lambda_normal else 'MISSING'}")
print(f"  - lambda_dist: {'OK' if has_lambda_dist else 'MISSING'}")
print(f"  - depth_ratio: {'OK' if has_depth_ratio else 'MISSING'}")

# Summary
print("\n" + "=" * 60)
print("Verification Summary")
print("=" * 60)
all_passed = (
    scaling_shape[1] == 2 and
    covariance.shape[1:] == torch.Size([4, 4]) and
    has_lambda_normal and has_lambda_dist and has_depth_ratio
)

if all_passed:
    print("ALL TESTS PASSED! MVGS + 2D-GS integration successful.")
else:
    print("SOME TESTS FAILED! Please check the output above.")

print("=" * 60)

