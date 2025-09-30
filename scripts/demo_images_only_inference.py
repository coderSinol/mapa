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
        
        # Convert to numpy arrays
        original_mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        valid_mask_np = valid_mask.cpu().numpy()
        mask = original_mask & valid_mask_np  # Combine with valid depth mask
        pts3d_np = pts3d_cam_torch.cpu().numpy()  # Use camera coordinates
        image_np = pred["img_no_norm"][0].cpu().numpy()

        # Save masks as images
        mask_save_dir = "/tmp/mapanything/masks"
        os.makedirs(mask_save_dir, exist_ok=True)
        
        # Save original mask (from model prediction)
        original_mask_img = Image.fromarray((original_mask * 255).astype(np.uint8))
        original_mask_img.save(f"{mask_save_dir}/view_{view_idx:03d}_original_mask.png")
        
        # Save valid mask (depth > 0)
        valid_mask_img = Image.fromarray((valid_mask_np * 255).astype(np.uint8))
        valid_mask_img.save(f"{mask_save_dir}/view_{view_idx:03d}_valid_mask.png")
        
        # Save combined mask
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
