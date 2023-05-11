import csv
import json
from pathlib import Path

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    # Convert quaternion to rotation matrix
    R = [
        [
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
        ],
        [
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qx * qw),
        ],
        [
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx * qx + qy * qy),
        ],
    ]
    return R

def convert_data(images_dir, depth_images_dir, poses_file, output_file):
    # Load poses
    poses = []
    with open(poses_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            print(row)
            poses.append(tuple(float(val) for val in row[1:]))

    frames = []
    for i, pose in enumerate(poses):
        px, py, pz, qx, qy, qz, qw = pose
        R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        T = [px, py, pz]

        transform_matrix = [
            [R[0][0], R[0][1], R[0][2], T[0]],
            [R[1][0], R[1][1], R[1][2], T[1]],
            [R[2][0], R[2][1], R[2][2], T[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]

        image_path = Path(images_dir) / f"image{i:03d}.png"
        frames.append(
            {
                "file_path": str(image_path),
                "transform_matrix": transform_matrix,
                "colmap_im_id": i,
            }
        )

    output_data = {
        "w": 640,
        "h": 480,
        "fl_x": 952.828,
        "fl_y": 952.828,
        "cx": 646.699,
        "cy": 342.637,
        "frames": frames,
    }

    # Save the output data as a JSON file
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    convert_data(
        images_dir="images",
        depth_images_dir="depths",
        poses_file="Positions.csv",
        output_file="transforms.json",
    )