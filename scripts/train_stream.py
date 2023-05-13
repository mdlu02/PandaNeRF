#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import os
import json

from common import srgb_to_linear
import pyngp as ngp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream", action="store_true", help="Stream images directly to InstantNGP."
    )
    parser.add_argument(
        "--n_frames",
        default=10,
        type=int,
        help="Number of frames before saving the dataset. Also used as the number of cameras to remember when streaming.",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to listen for files."
    )
    parser.add_argument(
        "--transforms_path",
        type=str,
        required=True,
        help="Path to read the transforms from."
    )
        
    return parser.parse_args()


def set_frame(
    testbed,
    frame_idx: int,
    rgb: np.ndarray,
    depth: np.ndarray,
    depth_scale: float,
    X_WV: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
):
    testbed.nerf.training.set_image(
        frame_idx=frame_idx,
        img=rgb,
        depth_img=depth,
        depth_scale=depth_scale * testbed.nerf.training.dataset.scale,
    )
    testbed.nerf.training.set_camera_extrinsics(
        frame_idx=frame_idx, camera_to_world=X_WV
    )
    testbed.nerf.training.set_camera_intrinsics(
        frame_idx=frame_idx, fx=fx, fy=fy, cx=cx, cy=cy
    )

class Frame:
    def __init__(self, image, width, height, has_depth, depth_image, depth_width, depth_height, transform_matrix, fl_x, fl_y, cx, cy):
        self.image = image
        self.width = width
        self.height = height
        self.has_depth = has_depth
        self.depth_image = depth_image
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.transform_matrix = transform_matrix
        self.fl_x = fl_x
        self.fl_y = fl_y
        self.cx = cx
        self.cy = cy


class DataReader:
    """
    Watches a directory for new images and reads them.
    """

    def __init__(self, path: str, transforms: list[list[list[float]]]):
        self.path = path
        self.image_index = 0
        self.transforms = transforms

    def _get_next_file(self):
        # Get file at image_index
        image_prefix = "test" # adjust as needed
        file_path = f"{self.path}/{image_prefix}{self.image_index:03}.png"
        return file_path if os.path.exists(file_path) and self.image_index <= 19 else None

    def read_next(self) -> Frame:
        # Check if there is a new file
        if (file_path := self._get_next_file()) is None:
            return None
        # Read image as RGB
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Replace with your camera's intrinsic parameters as needed
        frame = Frame(
            image=image,
            width=640,
            height=480,
            has_depth=False,
            depth_image=None,
            depth_width=0,
            depth_height=0,
            transform_matrix=self.transforms[self.image_index],
            fl_x=614.7780019980603,
            fl_y=606.6077841520959,
            cx=320.90852493306005,
            cy=229.1448299688645,
        )
        self.image_index += 1
        return frame


def live_streaming_loop(reader: DataReader, max_cameras: int):
    max_cameras -= 1
    # Start InstantNGP
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.init_window(1920, 1080)
    testbed.reload_network_from_file()
    # testbed.visualize_unit_cube = True
    testbed.nerf.visualize_cameras = True

    camera_index = 0  # Current camera index we are replacing in InstantNGP
    total_frames = 0  # Total frames received

    # Create Empty Dataset
    testbed.create_empty_nerf_dataset(max_cameras, aabb_scale=32)
    testbed.nerf.training.n_images_for_training = 0
    testbed.up_dir = np.array([1.0, 0.0, 0.0])

    # Start InstantNGP and Reader loop
    while testbed.frame():
        sample = reader.read_next()  # Get frame from NeRFCapture
        if sample:
            print(f"Frame {total_frames + 1} received")

            # RGB
            image = (
                np.asarray(sample.image, dtype=np.uint8)
                .reshape((sample.height, sample.width, 3))
                .astype(np.float32)
                / 255.0
            )
            image = np.concatenate(
                [image, np.zeros((sample.height, sample.width, 1), dtype=np.float32)],
                axis=-1,
            )

            # Depth if available
            depth = None
            if sample.has_depth:
                depth = (
                    np.asarray(sample.depth_image, dtype=np.uint8)
                    .view(dtype=np.float32)
                    .reshape((sample.depth_height, sample.depth_width))
                )
                depth = cv2.resize(
                    depth,
                    dsize=(sample.width, sample.height),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Transform
            X_WV = (
                np.asarray(sample.transform_matrix, dtype=np.float32).T
                .reshape((4, 4))
                .T[:3, :]
                .copy()
            )

            # Add frame to InstantNGP
            set_frame(
                testbed,
                frame_idx=camera_index,
                rgb=srgb_to_linear(image),
                depth=depth,
                depth_scale=1,
                X_WV=X_WV,
                fx=sample.fl_x,
                fy=sample.fl_y,
                cx=sample.cx,
                cy=sample.cy,
            )

            # Update index
            total_frames += 1
            testbed.nerf.training.n_images_for_training = min(total_frames, max_cameras)
            camera_index = (camera_index + 1) % max_cameras

            if total_frames == 1:
                testbed.first_training_view()
                # testbed.render_groundtruth = True
                
def get_transforms(transforms_path):
    """
    Reads the transforms from the transforms file and returns them as a list of 4x4 lists.
    The transforms file is a JSON file that has the following format:
    {
        "frames": [
            {
                "file_path": "./scripts/data/test000.png",
                "transform_matrix": [0.0, 0.0, 0.0, 0.0, ...],
            },
        ]
    }
    """
    index_to_transform = {}
    with open(transforms_path, "r") as f:
        data = json.load(f)
    for frame in data["frames"]:
        file_path = frame["file_path"]
        transform_matrix = frame["transform_matrix"]
        index = int(file_path.split("/")[-1].split(".")[0].split("test")[-1])
        index_to_transform[index] = transform_matrix
    # sort by index
    transforms = []
    for i in range(len(index_to_transform)):
        transforms.append(index_to_transform[i])
    return transforms


if __name__ == "__main__":
    args = parse_args()
    
    transforms = get_transforms(args.transforms_path)

    reader = DataReader(args.path, transforms)

    live_streaming_loop(reader, args.n_frames)
