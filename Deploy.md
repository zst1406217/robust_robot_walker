# Deploy the Policy on Your Real Unitree Robot

This guide demonstrates how to deploy the trained model on the **Unitree A1** robot (with **Nvidia Jetson NX**).

## Install Dependencies on A1

To deploy the trained model on the A1, follow these steps:

1. **Copy the Code**  
   Copy the `onboard_code` folder to the robot. Then, copy the checkpoint folder into it.

2. **Install ROS and the Unitree ROS Package**  
   Install ROS and the [Unitree ROS package](https://github.com/Tsinghua-MARS-Lab/unitree_ros_real.git). Follow the instructions to set up the robot on the `main` branch. Assume your ROS workspace is located in `~/unitree_ws`.

3. **Install PyTorch on Python 3 Environment**  
   Install PyTorch in a Python 3 environment (we recommend using conda for the virtual environment). Download the appropriate pip wheel file for PyTorch v1.10.0 from [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048). Then install it using:

   ```bash
   pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
   ```

4. **Install Other Dependencies and `rsl_rl`**  
   Install the required dependencies and the PPO implementation (`rsl_rl`):

   ```bash
   pip install numpy==1.16.4 numpy-ros
   pip install -e ./rsl_rl
   ```

## Run the Policy on A1

### 1. Power on the Robot
Put the robot on the ground, power it on, and set the robot into developer mode.

### 2. Launch Terminals
Launch 2 terminals on the robot (either through 2 SSH connections from your computer or another method), and name them **T_ros** and **T_policy**.

### 3. In Terminal `T_ros`:
Run the following commands to launch the ROS node for A1:

```bash
cd ~/unitree_ws
source devel/setup.bash
roslaunch unitree_legged_real robot.launch
```

This will start the ROS node. **Note:** The robot will not move unless `dryrun:=False` is specified. Leave `dryrun:=True` for testing, and set it to `dryrun:=False` only when you're ready to allow the robot to move.

### 4. In Terminal `T_policy`:
Run the following command to start the policy using the trained model:

```bash
cd onboard_code
python a1_ros_run.py --walkdir checkpoint
```

Here, `checkpoint` is the directory containing the trained model.

Once the ROS node is launched with `dryrun:=False`, the robot will stand up, and you can start the policy by pressing **R1** on the remote control. You should see the message:  
`Robot standing up procedure finished!`

### 5. Controls via Remote:
- **L-Y**: Move forward/backward  
- **L-X**: Move leftward/rightward  
- **R-X**: Turn left/right

### 6. Shutdown
To stop the script and the ROS node, press **R2** on the remote control. This can also be used in case of an emergency.

---