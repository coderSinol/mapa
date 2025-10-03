# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything Demo: Images-Only Inference with Visualization

Usage:
    python demo_images_only_inference.py --help
"""

import argparse
import json
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import rerun as rr
import torch
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.geometry import depthmap_to_camera_frame
from mapanything.utils.image import load_images
from mapanything.utils.viz import (
    predictions_to_glb,
    script_add_rerun_args,
)

# Import additional libraries for floor detection
try:
    from scipy import ndimage
    from sklearn.linear_model import RANSAC, LinearRegression
    import cv2
    FLOOR_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some packages for floor detection not available: {e}")
    print("Install with: pip install scipy scikit-learn opencv-python")
    FLOOR_DETECTION_AVAILABLE = False


def detect_floor_from_depth(depthmap_np, pts3d_np, valid_mask, method='ransac'):
    """
    Detect floor plane from depth map and 3D points
    
    Args:
        depthmap_np: Depth map as numpy array (H, W)
        pts3d_np: 3D points in camera coordinates (H, W, 3)
        valid_mask: Boolean mask for valid depth points (H, W)
        method: 'ransac', 'lowest_plane', or 'gradient'
    
    Returns:
        floor_mask: Boolean mask indicating floor pixels (H, W)
        floor_info: Dictionary with floor detection information
    """
    height, width = depthmap_np.shape
    floor_mask = np.zeros((height, width), dtype=bool)
    
    if method == 'ransac':
        # Use RANSAC to fit a plane to 3D points
        valid_points = pts3d_np[valid_mask]
        
        if len(valid_points) < 100:
            print("Not enough valid points for RANSAC")
            return floor_mask, {"method": method, "success": False}
        
        # Sample points for efficiency (use every Nth point)
        sample_step = max(1, len(valid_points) // 10000)
        sampled_points = valid_points[::sample_step]
        
        # Fit plane using RANSAC
        # Floor is typically the largest horizontal plane
        ransac = RANSAC(
            LinearRegression(),
            min_samples=3,
            residual_threshold=0.1,  # 10cm tolerance
            max_trials=1000
        )
        
        try:
            # Use XZ coordinates to fit Y (assuming Y is up/down in camera frame)
            X = sampled_points[:, [0, 2]]  # X and Z coordinates
            y = sampled_points[:, 1]       # Y coordinate
            
            ransac.fit(X, y)
            
            # Get plane parameters
            coef = ransac.estimator_.coef_
            intercept = ransac.estimator_.intercept_
            
            # Calculate distance of all valid points to the plane
            valid_indices = np.where(valid_mask)
            valid_pts_3d = pts3d_np[valid_indices]
            
            # Distance to plane: |ax + bz + c - y| / sqrt(a² + 1 + b²)
            a, b = coef[0], coef[1]
            c = intercept
            
            distances = np.abs(a * valid_pts_3d[:, 0] + b * valid_pts_3d[:, 2] + c - valid_pts_3d[:, 1])
            distances = distances / np.sqrt(a**2 + 1 + b**2)
            
            # Points close to the plane are floor candidates
            floor_threshold = 0.05  # 5cm tolerance
            floor_point_mask = distances < floor_threshold
            
            # Map back to image coordinates
            floor_indices = (valid_indices[0][floor_point_mask], valid_indices[1][floor_point_mask])
            floor_mask[floor_indices] = True
            
            # Post-process: remove small disconnected regions
            floor_mask = ndimage.binary_opening(floor_mask, structure=np.ones((3, 3)))
            floor_mask = ndimage.binary_closing(floor_mask, structure=np.ones((5, 5)))
            
            return floor_mask, {
                "method": method,
                "success": True,
                "plane_normal": np.array([a, -1, b]) / np.sqrt(a**2 + 1 + b**2),
                "plane_distance": c,
                "num_floor_pixels": np.sum(floor_mask)
            }
            
        except Exception as e:
            print(f"RANSAC failed: {e}")
            return floor_mask, {"method": method, "success": False, "error": str(e)}
    
    elif method == 'lowest_plane':
        # Simple heuristic: floor is typically at the bottom of the image with consistent depth
        # Look at bottom portion of image
        bottom_region = int(height * 0.7)  # Bottom 30% of image
        bottom_depths = depthmap_np[bottom_region:, :]
        bottom_valid = valid_mask[bottom_region:, :]
        
        if np.sum(bottom_valid) == 0:
            return floor_mask, {"method": method, "success": False}
        
        # Find median depth in bottom region
        bottom_depth_values = bottom_depths[bottom_valid]
        median_depth = np.median(bottom_depth_values)
        
        # Floor pixels should be within some range of this median depth
        depth_tolerance = 0.5  # 50cm tolerance
        depth_mask = np.abs(depthmap_np - median_depth) < depth_tolerance
        
        # Combine with valid mask
        floor_candidates = depth_mask & valid_mask
        
        # Focus on bottom half of image
        floor_candidates[:height//2, :] = False
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        floor_candidates_clean = cv2.morphologyEx(
            floor_candidates.astype(np.uint8), 
            cv2.MORPH_OPEN, 
            kernel
        )
        floor_candidates_clean = cv2.morphologyEx(
            floor_candidates_clean, 
            cv2.MORPH_CLOSE, 
            kernel
        )
        
        floor_mask = floor_candidates_clean.astype(bool)
        
        return floor_mask, {
            "method": method,
            "success": True,
            "median_depth": median_depth,
            "num_floor_pixels": np.sum(floor_mask)
        }
    
    elif method == 'gradient':
        # Use depth gradients - floor should have low gradient (relatively flat)
        # Calculate gradients
        grad_x = cv2.Sobel(depthmap_np, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depthmap_np, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Floor should have low gradient
        low_gradient_threshold = np.percentile(gradient_magnitude[valid_mask], 20)
        low_gradient_mask = gradient_magnitude < low_gradient_threshold
        
        # Combine with valid mask and focus on bottom portion
        floor_candidates = low_gradient_mask & valid_mask
        floor_candidates[:int(height * 0.5), :] = False  # Remove top half
        
        # Clean up with morphological operations
        kernel = np.ones((7, 7), np.uint8)
        floor_mask = cv2.morphologyEx(
            floor_candidates.astype(np.uint8), 
            cv2.MORPH_OPEN, 
            kernel
        ).astype(bool)
        
        return floor_mask, {
            "method": method,
            "success": True,
            "gradient_threshold": low_gradient_threshold,
            "num_floor_pixels": np.sum(floor_mask)
        }
    
    return floor_mask, {"method": method, "success": False}


def log_data_to_rerun(
    image, depthmap, pose, intrinsics, pts3d, mask, base_name, pts_name, viz_mask=None
):
    """Log visualization data to Rerun"""
    # Log camera info and loaded data
    height, width = image.shape[0], image.shape[1]
    rr.log(
        base_name,
        rr.Transform3D(
            translation=pose[:3, 3],
            mat3x3=pose[:3, :3],
        ),
    )
    rr.log(
        f"{base_name}/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=1.0,
        ),
    )
    rr.log(
        f"{base_name}/pinhole/rgb",
        rr.Image(image),
    )
    rr.log(
        f"{base_name}/pinhole/depth",
        rr.DepthImage(depthmap),
    )
    if viz_mask is not None:
        rr.log(
            f"{base_name}/pinhole/mask",
            rr.SegmentationImage(viz_mask.astype(int)),
        )

    # Log points in 3D
    filtered_pts = pts3d[mask]
    filtered_pts_col = image[mask]

    rr.log(
        pts_name,
        rr.Points3D(
            positions=filtered_pts.reshape(-1, 3),
            colors=filtered_pts_col.reshape(-1, 3),
        ),
    )


def get_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="MapAnything Demo: Visualize metric 3D reconstruction from images"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to folder containing images for reconstruction",
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory efficient inference for reconstruction (trades off speed)",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed model (facebook/map-anything-apache)",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="Enable visualization with Rerun",
    )
    parser.add_argument(
        "--save_glb",
        action="store_true",
        default=False,
        help="Save reconstruction as GLB file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.glb",
        help="Output path for GLB file (default: output.glb)",
    )
    parser.add_argument(
        "--detect_floor",
        action="store_true",
        default=True,
        help="Enable floor detection from depth maps",
    )

    return parser


def main():
    # Parser for arguments and Rerun
    parser = get_parser()
    script_add_rerun_args(
        parser
    )  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    # Get inference device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model from HuggingFace
    if args.apache:
        model_name = "facebook/map-anything-apache"
        print("Loading Apache 2.0 licensed MapAnything model...")
    else:
        model_name = "facebook/map-anything"
        print("Loading CC-BY-NC 4.0 licensed MapAnything model...")
    model = MapAnything.from_pretrained(model_name).to(device)

    # Load images
    print(f"Loading images from: {args.image_folder}")
    views = load_images(args.image_folder)
    print(f"Loaded {len(views)} views")

    # Run model inference
    print("Running inference...")
    outputs = model.infer(
        views, memory_efficient_inference=args.memory_efficient_inference
    )
    print("Inference complete!")

    # Save outputs to JSON file
    os.makedirs("/tmp/mapanything", exist_ok=True)
    # Convert outputs to serializable format
    serializable_outputs = []
    for i, output in enumerate(outputs):
        serializable_output = {}
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                serializable_output[key] = value.cpu().numpy().tolist()
            else:
                serializable_output[key] = value
        serializable_outputs.append(serializable_output)
    
    with open("/tmp/mapanything/outputs1.json", "w") as f:
        json.dump(serializable_outputs, f, indent=2)
    print("Outputs saved to /tmp/mapanything/outputs1.json")

    # Prepare lists for GLB export if needed
    world_points_list = []  # Will contain camera coordinates (pts3d_cam)
    images_list = []
    masks_list = []

    # Initialize Rerun if visualization is enabled
    if args.viz:
        print("Starting visualization...")
        viz_string = "MapAnything_Visualization"
        rr.script_setup(args, viz_string)
        rr.set_time("stable_time", sequence=0)
        rr.log("mapanything", rr.ViewCoordinates.RDF, static=True)

    # Loop through the outputs
    for view_idx, pred in enumerate(outputs):
        # Extract data from predictions
        depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
        intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
        camera_pose_torch = pred["camera_poses"][0]  # (4, 4)

        # Use camera coordinates - try pre-computed first, fallback to computing from depth
        if "pts3d_cam" in pred:
            # Use pre-computed camera coordinates from model output
            pts3d_cam_torch = pred["pts3d_cam"][0]  # (H, W, 3)
            valid_mask = depthmap_torch > 0.0
        else:
            # Compute camera coordinates from depth map and intrinsics
            pts3d_cam_torch, valid_mask = depthmap_to_camera_frame(
                depthmap_torch, intrinsics_torch
            )

        # Convert to numpy arrays for visualization
        # Check if mask key exists in pred, if not, fill with boolean trues in the size of depthmap_torch
        if "mask" in pred:
            original_mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        else:
            # Fill with boolean trues in the size of depthmap_torch
            original_mask = np.ones_like(depthmap_torch.cpu().numpy(), dtype=bool)
        
        # Convert to numpy arrays
        # original_mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        valid_mask_np = valid_mask.cpu().numpy()
        mask = original_mask & valid_mask_np  # Combine with valid depth mask
        pts3d_np = pts3d_cam_torch.cpu().numpy()  # Use camera coordinates
        image_np = pred["img_no_norm"][0].cpu().numpy()

        # Save depth map as image
        depth_save_dir = "/tmp/mapanything/depth_maps"
        os.makedirs(depth_save_dir, exist_ok=True)
        
        # Convert depth map to numpy and normalize for visualization
        depthmap_np = depthmap_torch.cpu().numpy()
        
        # Normalize depth values to 0-255 range for saving as image
        # Handle case where all depth values might be the same
        if depthmap_np.max() > depthmap_np.min():
            depth_normalized = (depthmap_np - depthmap_np.min()) / (depthmap_np.max() - depthmap_np.min())
        else:
            depth_normalized = np.zeros_like(depthmap_np)
        
        depth_img = (depth_normalized * 255).astype(np.uint8)
        depth_pil = Image.fromarray(depth_img, mode='L')  # 'L' mode for grayscale
        depth_pil.save(f"{depth_save_dir}/view_{view_idx:03d}_depth.png")
        
        # Also save raw depth values as numpy file for precise reconstruction
        np.save(f"{depth_save_dir}/view_{view_idx:03d}_depth_raw.npy", depthmap_np)
        
        print(f"Saved depth map for view {view_idx} - Range: [{depthmap_np.min():.3f}, {depthmap_np.max():.3f}]")

        # Detect floor area from depth map
        if args.detect_floor and FLOOR_DETECTION_AVAILABLE:
            print(f"Detecting floor for view {view_idx}...")
            
            # Try different methods for floor detection
            methods = ['ransac', 'lowest_plane', 'gradient']
            floor_results = {}
            
            for method in methods:
                floor_mask, floor_info = detect_floor_from_depth(depthmap_np, pts3d_np, valid_mask_np, method=method)
                floor_results[method] = {'mask': floor_mask, 'info': floor_info}
                
                if floor_info['success']:
                    print(f"  {method}: Success - {floor_info.get('num_floor_pixels', 0)} floor pixels")
                else:
                    print(f"  {method}: Failed")
            
            # Save floor detection results
            floor_save_dir = "/tmp/mapanything/floor_detection"
            os.makedirs(floor_save_dir, exist_ok=True)
            
            for method, result in floor_results.items():
                if result['info']['success']:
                    # Save floor mask as image
                    floor_mask_img = (result['mask'] * 255).astype(np.uint8)
                    floor_pil = Image.fromarray(floor_mask_img, mode='L')
                    floor_pil.save(f"{floor_save_dir}/view_{view_idx:03d}_floor_{method}.png")
                    
                    # Create colored overlay on original image for visualization
                    overlay = image_np.copy()
                    floor_overlay = result['mask'][:, :, np.newaxis] * np.array([0, 255, 0])  # Green overlay
                    overlay = np.where(result['mask'][:, :, np.newaxis], 
                                     0.7 * overlay + 0.3 * floor_overlay, 
                                     overlay).astype(np.uint8)
                    overlay_pil = Image.fromarray(overlay)
                    overlay_pil.save(f"{floor_save_dir}/view_{view_idx:03d}_floor_{method}_overlay.png")
            
            # Save floor detection info as JSON
            floor_info_path = f"{floor_save_dir}/view_{view_idx:03d}_floor_info.json"
            serializable_floor_info = {}
            for method, result in floor_results.items():
                info_copy = result['info'].copy()
                # Convert numpy arrays to lists for JSON serialization
                for key, value in info_copy.items():
                    if isinstance(value, np.ndarray):
                        info_copy[key] = value.tolist()
                serializable_floor_info[method] = info_copy
            
            with open(floor_info_path, 'w') as f:
                json.dump(serializable_floor_info, f, indent=2)
        elif args.detect_floor and not FLOOR_DETECTION_AVAILABLE:
            print(f"Skipping floor detection for view {view_idx} - required packages not available")

        # Save masks as images
        mask_save_dir = "/tmp/mapanything/masks"
        os.makedirs(mask_save_dir, exist_ok=True)
        
        # Debug: Print mask statistics
        print(f"View {view_idx} - Original mask: {np.sum(original_mask)} True pixels out of {original_mask.size} total")
        print(f"View {view_idx} - Valid mask: {np.sum(valid_mask_np)} True pixels out of {valid_mask_np.size} total")
        print(f"View {view_idx} - Combined mask: {np.sum(mask)} True pixels out of {mask.size} total")
        
        # Save original mask (from model prediction) - convert boolean to 0/255
        original_mask_img = Image.fromarray((original_mask * 255).astype(np.uint8))
        original_mask_img.save(f"{mask_save_dir}/view_{view_idx:03d}_original_mask.png")
        
        # Save valid mask (depth > 0) - convert boolean to 0/255
        valid_mask_img = Image.fromarray((valid_mask_np * 255).astype(np.uint8))
        valid_mask_img.save(f"{mask_save_dir}/view_{view_idx:03d}_valid_mask.png")
        
        # Save combined mask - convert boolean to 0/255
        combined_mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        combined_mask_img.save(f"{mask_save_dir}/view_{view_idx:03d}_combined_mask.png")
        
        print(f"Saved masks for view {view_idx} to {mask_save_dir}")

        # Store data for GLB export if needed
        if args.save_glb:
            world_points_list.append(pts3d_np)  # Now contains camera coordinates
            images_list.append(image_np)
            masks_list.append(mask)

        # Log to Rerun if visualization is enabled
        if args.viz:
            log_data_to_rerun(
                image=image_np,
                depthmap=depthmap_torch.cpu().numpy(),
                pose=camera_pose_torch.cpu().numpy(),
                intrinsics=intrinsics_torch.cpu().numpy(),
                pts3d=pts3d_np,
                mask=mask,
                base_name=f"mapanything/view_{view_idx}",
                pts_name=f"mapanything/pointcloud_view_{view_idx}",
                viz_mask=mask,
            )

    if args.viz:
        print("Visualization complete! Check the Rerun viewer.")

    # Export GLB if requested
    if args.save_glb:
        print(f"Saving GLB file to: {args.output_path}")

        # Stack all views
        camera_points = np.stack(world_points_list, axis=0)  # Now contains camera coordinates
        images = np.stack(images_list, axis=0)
        final_masks = np.stack(masks_list, axis=0)

        # Convert camera_points to vertices_3d format and save as JSON
        vertices_3d = camera_points.reshape(-1, 3)
        camera_points_json_path = "/tmp/mapanything/camera_points.json"
        camera_points_data = {
            "vertices_3d": vertices_3d.tolist(),
            "metadata": {
                "original_shape": list(camera_points.shape),
                "vertices_shape": list(vertices_3d.shape),
                "num_views": camera_points.shape[0],
                "total_vertices": vertices_3d.shape[0],
                "coordinate_system": "camera",
                "description": "Flattened 3D vertices from all views in camera coordinates, shape: (total_points, 3)"
            }
        }
        with open(camera_points_json_path, 'w') as f:
            json.dump(camera_points_data, f, indent=2)
        print(f"Camera coordinates saved to: {camera_points_json_path}")

        # Create predictions dict for GLB export
        predictions = {
            "world_points": camera_points,  # Keep the key name for compatibility with predictions_to_glb
            "images": images,
            "final_masks": final_masks,
        }

        # Convert to GLB scene
        scene_3d = predictions_to_glb(predictions, as_mesh=True)

        # Save GLB file
        scene_3d.export(args.output_path)
        print(f"Successfully saved GLB file: {args.output_path}")
    else:
        print("Skipping GLB export (--save_glb not specified)")


if __name__ == "__main__":
    main()
