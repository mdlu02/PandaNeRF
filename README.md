# GraspNeRF

Here is the code for our CSCI 2952-0 project. Along with our actual code to move the arm and train the NeRF in real time, a significant amount of effort went into setting up the necessary dependencies and packages on the PC in the robotics lab. The same code can be found on the PC in the panda_ws directory under motion_planning. We have also included our CAD and .stl files for the realsense mounts we tried.

## Live training
We provide code in order to train InstantNGP in real time as data comes in from the robot. We also facilitate capture on one machine and training on another, remote machine.

### Dependencies
In order to use the live streamed training pipeline, you will need to clone the [InstantNGP repository](https://github.com/NVlabs/instant-ngp) and have the following packages installed:
- `pyngp` from InstantNGP
- `opencv`
- `numpy`

### AWS S3
You also need to set up the AWS CLI on both machines (you could also run capture and training on the same machine), and you must create an S3 bucket. Then populate the `S3_BUCKET` variable in [listen.sh](./scripts/listen.sh) and any calls to `upload_to_s3` in [grasp_nerf.py](./scripts/grasp_nerf.py) with your bucket name.

### Running
Then all is needed is to run `listen.sh` and `train_stream.py` on the training machine and begin the capture process on the other machine.