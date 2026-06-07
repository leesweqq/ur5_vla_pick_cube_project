import argparse
import math
import shutil
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# =========================================================
# Camera setup
# =========================================================
# Define the main GUI window size used when running PyBullet in visual mode.
GUI_WINDOW_SIZE = (1600, 1000)
# Define the human-facing GUI camera so the scene is easy to inspect.
GUI_CAMERA_VIEW = {
    "distance": 1.0,
    "yaw": 90,
    "pitch": -30,
    "target": [0.50, 0.00, 0.85],
}

# Define three observation cameras used as visual inputs for data collection.
OBSERVATION_CAMERA_SPECS = (
    {
        "eye": [1.05, -0.55, 1.05],
        "target": [0.50, 0.00, 0.78],
        "up": [0, 0, 1],
        "fov": 55,
    },
    {
        "eye": [0.75, 0.00, 1.45],
        "target": [0.50, 0.00, 0.78],
        "up": [0, 0, 1],
        "fov": 50,
    },
    {
        "eye": [0.20, 0.75, 1.00],
        "target": [0.50, 0.00, 0.78],
        "up": [0, 0, 1],
        "fov": 55,
    },
)


# =========================================================
# Robot wrapper
# =========================================================
class UR5Robotiq85:
    """UR5 arm with a Robotiq 85 gripper."""

    def __init__(self, pos, ori):
        # Store the robot base pose and convert Euler orientation to a PyBullet quaternion.
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)

        # Cache robot and gripper configuration values used throughout control.
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.gripper_range = [0.0, 0.085]
        self.max_velocity = 3.0
        self.id = None

    def load(self):
        """Load the URDF and configure the gripper mimic joints."""
        # Load the robot model as a fixed-base manipulator.
        self.id = p.loadURDF(
            "./urdf/ur5_robotiq_85.urdf",
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
        )
        # Build joint metadata first, then configure mimic joints for the gripper.
        self._parse_joint_info()
        self.__setup_mimic_joints__()

        # Increase finger friction so the cube is less likely to slip during grasping.
        for link_id in [12, 17]:
            p.changeDynamics(
                self.id,
                link_id,
                lateralFriction=1000.0,
                spinningFriction=1.0,
                frictionAnchor=1,
            )

    def _parse_joint_info(self):
        """Read PyBullet joint metadata and cache controllable joints."""
        joint_info = namedtuple(
            "jointInfo",
            [
                "id",
                "name",
                "type",
                "lowerLimit",
                "upperLimit",
                "maxForce",
                "maxVelocity",
                "controllable",
            ],
        )
        # Store all joints and the subset that can be controlled by motors.
        self.joints = []
        self.controllable_joints = []

        # Iterate through PyBullet joints and extract useful control limits and names.
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            joint_lower_limit = info[8]
            joint_upper_limit = info[9]
            joint_max_force = info[10]
            joint_max_velocity = info[11]
            controllable = joint_type != p.JOINT_FIXED

            if controllable:
                self.controllable_joints.append(joint_id)

            self.joints.append(
                joint_info(
                    joint_id,
                    joint_name,
                    joint_type,
                    joint_lower_limit,
                    joint_upper_limit,
                    joint_max_force,
                    joint_max_velocity,
                    controllable,
                )
            )

        # Split the first six controllable joints as the UR5 arm joints.
        self.arm_controllable_joints = self.controllable_joints[: self.arm_num_dofs]
        self.arm_lower_limits = [
            joint.lowerLimit for joint in self.joints if joint.controllable
        ][: self.arm_num_dofs]
        self.arm_upper_limits = [
            joint.upperLimit for joint in self.joints if joint.controllable
        ][: self.arm_num_dofs]
        self.arm_joint_ranges = [
            upper - lower
            for lower, upper in zip(self.arm_lower_limits, self.arm_upper_limits)
        ]

    def __setup_mimic_joints__(self):
        """Tie the Robotiq child joints to the parent finger joint."""
        # The parent finger joint drives all child mimic joints through gear constraints.
        mimic_parent_name = "finger_joint"
        mimic_children_names = {
            "right_outer_knuckle_joint": 1,
            "left_inner_knuckle_joint": 1,
            "right_inner_knuckle_joint": 1,
            "left_inner_finger_joint": -1,
            "right_inner_finger_joint": -1,
        }
        self.mimic_parent_id = [
            joint.id for joint in self.joints if joint.name == mimic_parent_name
        ][0]
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name]
            for joint in self.joints
            if joint.name in mimic_children_names
        }

        # Create gear constraints so all gripper fingers move together.
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            constraint_id = p.createConstraint(
                self.id,
                self.mimic_parent_id,
                self.id,
                joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            p.changeConstraint(
                constraint_id,
                gearRatio=-multiplier,
                maxForce=100,
                erp=1,
            )

    def move_arm_ik_absolute(self, target_pos, target_orn):
        """Move the arm end effector to an absolute pose with IK."""
        # Use inverse kinematics to convert the target end-effector pose into joint targets.
        joint_poses = p.calculateInverseKinematics(
            self.id,
            self.eef_id,
            target_pos,
            target_orn,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
        )
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.id,
                joint_id,
                p.POSITION_CONTROL,
                joint_poses[i],
                maxVelocity=self.max_velocity,
            )

    def move_gripper(self, open_length):
        """Command the gripper opening length in meters."""
        # Clamp the requested opening and convert it into the Robotiq joint angle.
        open_length = np.clip(open_length, self.gripper_range[0], self.gripper_range[1])
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        p.setJointMotorControl2(
            self.id,
            self.mimic_parent_id,
            p.POSITION_CONTROL,
            targetPosition=open_angle,
        )

    def get_ee_pose(self):
        """Return end-effector position and orientation."""
        state = p.getLinkState(self.id, self.eef_id)
        pos = np.array(state[0], dtype=np.float32)
        orn = np.array(state[1], dtype=np.float32)
        return pos, orn

    def get_joint_positions(self):
        """Return 6 arm joints plus the gripper parent joint."""
        values = []
        for joint_id in self.arm_controllable_joints:
            values.append(p.getJointState(self.id, joint_id)[0])
        values.append(p.getJointState(self.id, self.mimic_parent_id)[0])
        return np.array(values, dtype=np.float32)


# =========================================================
# Environment
# =========================================================
class UR5PickCubeEnv:
    """PyBullet pick-cube environment for UR5, Robotiq 85, and SmolVLA."""

    def __init__(
        self,
        gui=False,
        image_size=(224, 224),
        seed=0,
        observation_camera_count=3,
        gui_window_size=GUI_WINDOW_SIZE,
    ):
        # Validate that the environment exposes between one and three observation cameras.
        if observation_camera_count < 1 or observation_camera_count > 3:
            raise ValueError("observation_camera_count must be between 1 and 3")

        # Store runtime configuration and initialize simulation object handles.
        self.gui = gui
        self.image_size = image_size
        self.observation_camera_count = observation_camera_count
        self.gui_window_size = gui_window_size
        self.physics_client = None
        self.robot = None
        self.cube_id = None
        self.table_id = None
        self.rng = np.random.default_rng(seed)

        # Use a fixed red cube so the language task and visual target are consistent.
        self.cube_color_name = "red"
        self.cube_color = [1.0, 0.0, 0.0, 1.0]

        # Language templates are sampled per episode to diversify task text.
        self.task_templates = [
            ("pick up the red cube", "pick up the red cube"),
            ("grasp the red cube", "grasp the red cube"),
            ("lift the red cube", "lift the red cube"),
            ("pick up the red cube with the gripper", "pick up the red cube with the gripper"),
            ("pick up the red cube on the table", "pick up the red cube on the table"),
            ("please pick up the red cube", "please pick up the red cube"),
            ("lift the red block", "lift the red block"),
            ("grasp and lift the red cube", "grasp and lift the red cube"),
            ("pick up this red cube", "pick up this red cube"),
            ("grasp and raise the red cube", "grasp and raise the red cube"),
        ]
        self.language_instruction = ""
        self.language_instruction_zh = ""
        self.language_instruction_en = ""

        # Control thresholds and fixed end-effector orientation used during grasping.
        self.fixed_ee_euler = [0.0, 1.57, 0.0]
        self.success_cube_height = 0.72
        self.min_ee_z = 0.772

        # Reset per-episode grasp state before loading objects.
        self.has_closed_gripper = False
        self.cube_spawn_pos = None
        self.cube_spawn_yaw = 0.0

    def connect(self):
        """Connect to PyBullet in GUI or DIRECT mode."""
        # Only disconnect when a physics client is currently active.
        if self.physics_client is not None:
            return

        # Select GUI mode for visualization or DIRECT mode for faster headless runs.
        if self.gui:
            width, height = self.gui_window_size
            options = f"--width={width} --height={height}"
            self.physics_client = p.connect(p.GUI, options=options)
            self.configure_gui_view()
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Add PyBullet's default asset path for plane, table, and cube URDF files.
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def disconnect(self):
        """Disconnect from PyBullet."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None

    def configure_gui_view(self):
        """Use one large human-facing PyBullet view and hide side previews."""
        if not self.gui:
            return

        # Disable small preview panels so the GUI shows one clean main viewport.
        preview_flags = [
            getattr(p, "COV_ENABLE_RGB_BUFFER_PREVIEW", None),
            getattr(p, "COV_ENABLE_DEPTH_BUFFER_PREVIEW", None),
            getattr(p, "COV_ENABLE_SEGMENTATION_MARK_PREVIEW", None),
        ]
        for flag in preview_flags:
            if flag is not None:
                p.configureDebugVisualizer(flag, 0)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=GUI_CAMERA_VIEW["distance"],
            cameraYaw=GUI_CAMERA_VIEW["yaw"],
            cameraPitch=GUI_CAMERA_VIEW["pitch"],
            cameraTargetPosition=GUI_CAMERA_VIEW["target"],
        )

    def step_sim(self, steps=2):
        """Advance simulation and keep GUI playback close to real time."""
        # Step physics repeatedly; GUI mode sleeps to keep playback readable.
        for _ in range(steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)

    def sample_cube_pose(self):
        """Sample a tabletop cube pose inside the robot workspace."""
        # Randomize the cube within a safe tabletop workspace reachable by the robot.
        x = self.rng.uniform(0.42, 0.58)
        y = self.rng.uniform(-0.12, 0.12)
        z = 0.65
        yaw = 0.0
        return np.array([x, y, z], dtype=np.float32), float(yaw)

    def sample_language_instruction(self):
        """Sample an English task instruction pair."""
        # Select one task instruction template for the current episode.
        index = int(self.rng.integers(len(self.task_templates)))
        return self.task_templates[index]

    def reset(self):
        """Reset the world, robot, cube, and task instruction."""
        # Start from a clean PyBullet world for every episode.
        self.connect()
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        self.configure_gui_view()
        p.setGravity(0, 0, -9.8)

        self.has_closed_gripper = False

        # Load the ground plane and table before adding the robot and cube.
        p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF(
            "table/table.urdf",
            [0.5, 0, 0],
            p.getQuaternionFromEuler([0, 0, 0]),
        )

        # Create and load the UR5 robot at the table height.
        self.robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
        self.robot.load()

        # Move the arm to a stable home posture before spawning the cube.
        home_joints = [0.0, -1.40, 1.45, -1.60, -1.57, 0.0]
        for i, joint_id in enumerate(self.robot.arm_controllable_joints):
            p.setJointMotorControl2(
                self.robot.id,
                joint_id,
                p.POSITION_CONTROL,
                home_joints[i],
            )

        self.robot.move_gripper(0.085)
        self.step_sim(30)

        # Sample cube pose and language instruction for this episode.
        self.cube_spawn_pos, self.cube_spawn_yaw = self.sample_cube_pose()
        self.language_instruction_zh, self.language_instruction_en = (
            self.sample_language_instruction()
        )
        self.language_instruction = self.language_instruction_en

        # Spawn the target cube and color it red.
        self.cube_id = p.loadURDF(
            "cube_small.urdf",
            self.cube_spawn_pos,
            p.getQuaternionFromEuler([0, 0, self.cube_spawn_yaw]),
        )
        p.changeVisualShape(self.cube_id, -1, rgbaColor=self.cube_color)

        self.step_sim(30)

        # Stabilize the cube pose after initial simulation steps.
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        p.resetBasePositionAndOrientation(
            self.cube_id,
            [cube_pos[0], cube_pos[1], cube_pos[2]],
            p.getQuaternionFromEuler([0, 0, self.cube_spawn_yaw]),
        )
        p.resetBaseVelocity(self.cube_id, [0, 0, 0], [0, 0, 0])

        self.configure_gui_view()
        self.step_sim(30)
        return self.get_obs()

    def render_camera(self, eye, target, up=(0, 0, 1), fov=55):
        """Render one RGB observation camera."""
        # Build PyBullet camera matrices and render an RGB image.
        width, height = self.image_size

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=target,
            cameraUpVector=up,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=float(width) / height,
            nearVal=0.01,
            farVal=3.0,
        )

        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER,
        )
        rgb = np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8)
        return rgb

    def get_camera1_image(self):
        """Return the front-side observation camera."""
        spec = OBSERVATION_CAMERA_SPECS[0]
        return self.render_camera(**spec)

    def get_camera2_image(self):
        """Return the overhead observation camera."""
        spec = OBSERVATION_CAMERA_SPECS[1]
        return self.render_camera(**spec)

    def get_camera3_image(self):
        """Return the opposite-side observation camera."""
        spec = OBSERVATION_CAMERA_SPECS[2]
        return self.render_camera(**spec)

    def capture_observation_images(self):
        """Capture observation cameras while preserving camera1-3 keys."""
        # Always return camera1-camera3 keys; duplicate the last available image if needed.
        images = {}
        last_image = None

        for index, spec in enumerate(OBSERVATION_CAMERA_SPECS, start=1):
            key = f"image_camera{index}"
            if index <= self.observation_camera_count:
                last_image = self.render_camera(**spec)
                images[key] = last_image
            else:
                images[key] = last_image.copy()

        return images

    def get_obs(self):
        """Return images, robot state, cube pose, and task text."""
        # Read robot state, camera images, cube position, and task text into one observation.
        ee_pos, ee_orn = self.robot.get_ee_pose()
        joint_pos = self.robot.get_joint_positions()
        images = self.capture_observation_images()

        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        cube_pos = np.array(cube_pos, dtype=np.float32)

        # Keep an expanded debug state for inspection while training uses joint positions.
        debug_state = np.concatenate([joint_pos, ee_pos, cube_pos], axis=0).astype(
            np.float32
        )

        return {
            "image_camera1": images["image_camera1"],
            "image_camera2": images["image_camera2"],
            "image_camera3": images["image_camera3"],
            "joint_positions": joint_pos,
            "ee_pos": ee_pos,
            "ee_orn": ee_orn,
            "cube_pos": cube_pos,
            "observation_state": debug_state,
            "task_zh": self.language_instruction_zh,
            "task_en": self.language_instruction_en,
        }

    def gripper_close(self):
        """Close the gripper until both fingers contact the cube or fully close."""
        # Start from the maximum opening and gradually close until both fingers contact the cube.
        grip_value = self.robot.gripper_range[1]
        left_finger_link = 12
        right_finger_link = 17
        force_threshold = 2.0
        max_iters = 120

        # Repeatedly check contact force while tightening the gripper.
        for _ in range(max_iters):
            contact_points = p.getContactPoints(bodyA=self.robot.id, bodyB=self.cube_id)

            left_force = 0.0
            right_force = 0.0

            # Track the strongest contact force on each gripper finger.
            for contact in contact_points:
                robot_link = contact[3]
                normal_force = contact[9]

                if robot_link == left_finger_link:
                    left_force = max(left_force, normal_force)
                elif robot_link == right_finger_link:
                    right_force = max(right_force, normal_force)

            if left_force > force_threshold and right_force > force_threshold:
                print(f"[Grasped] left={left_force:.2f}, right={right_force:.2f}")
                return True

            if grip_value <= self.robot.gripper_range[0]:
                break

            grip_value -= 0.001
            self.robot.move_gripper(grip_value)
            for _ in range(30):
                p.stepSimulation()

        return False

    def apply_action(self, action):
        """Apply a 7D delta action: dx, dy, dz, droll, dpitch, dyaw, gripper."""
        # Decode the 7D action into position delta, ignored rotation delta, and gripper command.
        dx, dy, dz, _droll, _dpitch, _dyaw, gripper = action

        ee_pos, _ = self.robot.get_ee_pose()
        target_orn = p.getQuaternionFromEuler(self.fixed_ee_euler)

        # Apply the position delta while preventing the end effector from going too low.
        target_pos = ee_pos + np.array([dx, dy, dz], dtype=np.float32)
        target_pos[2] = max(target_pos[2], self.min_ee_z)

        self.robot.move_arm_ik_absolute(target_pos.tolist(), target_orn)

        # Interpret gripper values above the threshold as open, otherwise close or hold closed.
        if gripper > 0.02:
            self.robot.move_gripper(0.085)
            self.step_sim(5)
            self.has_closed_gripper = False
        else:
            if not self.has_closed_gripper:
                grasped = self.gripper_close()
                self.has_closed_gripper = grasped
            else:
                self.step_sim(5)

        # Produce the standard Gym-like return values for the next control step.
        obs = self.get_obs()
        task_success = self.is_success()
        done = task_success or self.is_failure()
        reward = 1.0 if task_success else 0.0

        return obs, reward, done, {}

    def is_success(self):
        """A grasp is successful once the cube is lifted above the threshold."""
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        return cube_pos[2] > self.success_cube_height

    def is_failure(self):
        """Fail if the cube falls below the table area."""
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        return cube_pos[2] < 0.3


# =========================================================
# Expert policy
# =========================================================
def make_expert_action(env, phase):
    """Generate one scripted expert action for the current phase."""
    # Compare the robot and cube state against each phase completion condition.
    ee_pos, _ = env.robot.get_ee_pose()
    cube_pos, _ = p.getBasePositionAndOrientation(env.cube_id)
    cube_pos = np.array(cube_pos, dtype=np.float32)

    # Define waypoint heights for each phase of the scripted pick trajectory.
    approach_height = 0.84
    pregrasp_height = 0.805
    grasp_height = 0.782
    lift_height = 0.90

    # Add small noise so collected demonstrations are not perfectly identical.
    xy_noise = np.random.uniform(-0.003, 0.003, size=2).astype(np.float32)
    z_noise = np.random.uniform(-0.0015, 0.0015, size=1).astype(np.float32)

    # Choose a phase-specific target pose and gripper command.
    if phase == "approach_cube":
        target = np.array(
            [cube_pos[0] + xy_noise[0], cube_pos[1] + xy_noise[1], approach_height],
            dtype=np.float32,
        )
        gripper = 0.085

    elif phase == "pre_descend":
        target = np.array(
            [cube_pos[0] + xy_noise[0], cube_pos[1] + xy_noise[1], pregrasp_height],
            dtype=np.float32,
        )
        gripper = 0.085

    elif phase == "descend":
        target = np.array(
            [cube_pos[0], cube_pos[1], grasp_height + z_noise[0]],
            dtype=np.float32,
        )
        gripper = 0.085

    elif phase == "close":
        target = ee_pos.copy()
        gripper = 0.0

    elif phase == "hold_close":
        target = ee_pos.copy()
        gripper = 0.0

    elif phase == "lift":
        target = np.array(
            [cube_pos[0], cube_pos[1], lift_height],
            dtype=np.float32,
        )
        gripper = 0.0

    else:
        raise ValueError(phase)

    # Convert the absolute target into a clipped delta action for the environment.
    delta_pos = target - ee_pos
    delta_pos = np.clip(delta_pos, -0.015, 0.015)

    return np.array(
        [
            delta_pos[0],
            delta_pos[1],
            delta_pos[2],
            0.0,
            0.0,
            0.0,
            gripper,
        ],
        dtype=np.float32,
    )


def phase_done(env, phase, phase_step_count):
    """Return True when the scripted expert can move to the next phase."""
    ee_pos, _ = env.robot.get_ee_pose()
    cube_pos, _ = p.getBasePositionAndOrientation(env.cube_id)
    cube_pos = np.array(cube_pos, dtype=np.float32)

    if phase == "approach_cube":
        target = np.array([cube_pos[0], cube_pos[1], 0.84], dtype=np.float32)
        return np.linalg.norm(ee_pos - target) < 0.025

    if phase == "pre_descend":
        target = np.array([cube_pos[0], cube_pos[1], 0.805], dtype=np.float32)
        return np.linalg.norm(ee_pos - target) < 0.02

    if phase == "descend":
        target = np.array([cube_pos[0], cube_pos[1], 0.782], dtype=np.float32)
        return np.linalg.norm(ee_pos - target) < 0.012

    if phase == "close":
        return env.has_closed_gripper or phase_step_count >= 6

    if phase == "hold_close":
        return phase_step_count >= 6

    if phase == "lift":
        return env.is_success()

    return False


# =========================================================
# LeRobot dataset builder
# =========================================================
def build_lerobot_dataset(repo_id, root, image_size=(224, 224), fps=10):
    """Create the LeRobot dataset schema for three RGB cameras and 7D actions."""
    # Match dataset video feature shapes to the camera image resolution.
    width, height = image_size

    # Define the LeRobot schema: three RGB videos, robot state, and 7D action.
    features = {
        "observation.images.camera1": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.camera2": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.camera3": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "joint_0",
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "gripper",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        },
    }

    # Create a video-backed local LeRobot dataset with asynchronous image encoding.
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        features=features,
        use_videos=True,
        image_writer_processes=2,
        image_writer_threads=8,
        batch_encoding_size=1,
        streaming_encoding=False,
    )
    return dataset


# =========================================================
# Data collection
# =========================================================
def collect_dataset_lerobot(
    repo_id="local/ur5_pick_red_cube_3cam",
    root="./lerobot_dataset",
    num_episodes=100,
    gui=True,
    seed=42,
):
    """Collect successful expert episodes into a three-camera LeRobot dataset."""
    # Resolve the dataset root and remove old data to avoid mixing runs.
    root_path = Path(root)

    if root_path.exists():
        print(f"[INFO] remove existing dataset folder: {root_path.resolve()}")
        shutil.rmtree(root_path)

    # Open a short GUI-only preview without writing dataset frames.
    env = UR5PickCubeEnv(
        gui=gui,
        image_size=(224, 224),
        seed=seed,
        observation_camera_count=3,
    )
    dataset = build_lerobot_dataset(
        repo_id=repo_id,
        root=root,
        image_size=(224, 224),
        fps=10,
    )

    # Track how many episodes are saved versus discarded.
    saved_count = 0
    dropped_count = 0

    try:
        # Collect one full scripted demonstration per episode.
        for episode_index in range(num_episodes):
            obs = env.reset()

            # The scripted policy moves through fixed phases from approach to lift.
            phases = [
                "approach_cube",
                "pre_descend",
                "descend",
                "close",
                "hold_close",
                "lift",
            ]
            phase_idx = 0
            phase_step_count = 0
            success = False

            # Limit each episode length so failed attempts do not run forever.
            for _ in range(320):
                phase = phases[phase_idx]
                action = make_expert_action(env, phase)
                task = obs["task_zh"] if env.rng.random() < 0.5 else obs["task_en"]

                # Add the current observation, action, and task text to the episode buffer.
                frame = {
                    "observation.images.camera1": obs["image_camera1"],
                    "observation.images.camera2": obs["image_camera2"],
                    "observation.images.camera3": obs["image_camera3"],
                    "observation.state": obs["joint_positions"],
                    "action": action.astype(np.float32),
                    "task": task,
                }
                dataset.add_frame(frame)

                obs, reward, done, _ = env.apply_action(action)
                phase_step_count += 1

                # Advance the finite-state expert policy when the current phase is complete.
                if phase_done(env, phase, phase_step_count):
                    phase_idx += 1
                    phase_step_count = 0
                    if phase_idx >= len(phases):
                        done = True

                if done:
                    success = env.is_success()
                    break

            # Save only successful demonstrations; discard failed buffered frames.
            if success:
                dataset.save_episode()
                saved_count += 1
                print(
                    f"[Episode {episode_index:03d}] saved "
                    f"success={success} "
                    f"task='{env.language_instruction_en}' "
                    f"cube_pos={np.round(env.cube_spawn_pos, 4).tolist()} "
                    f"yaw={round(env.cube_spawn_yaw, 4)} "
                    f"(saved={saved_count}, dropped={dropped_count})"
                )
            else:
                dataset.clear_episode_buffer(delete_images=True)
                dropped_count += 1
                print(
                    f"[Episode {episode_index:03d}] dropped "
                    f"success={success} "
                    f"task='{env.language_instruction_en}' "
                    f"cube_pos={np.round(env.cube_spawn_pos, 4).tolist()} "
                    f"yaw={round(env.cube_spawn_yaw, 4)} "
                    f"(saved={saved_count}, dropped={dropped_count})"
                )

    finally:
        # Finalize dataset metadata and release simulation resources.
        dataset.finalize()
        env.disconnect()
        print("\n=========================================================")
        print(f"[FINAL] saved={saved_count}, dropped={dropped_count}")
        print("=========================================================")


def preview_environment(seed=42, seconds=20.0):
    """Open the GUI with one human-facing view for quick inspection."""
    env = UR5PickCubeEnv(
        gui=True,
        image_size=(224, 224),
        seed=seed,
        observation_camera_count=1,
    )
    try:
        env.reset()
        for _ in range(int(seconds * 240)):
            env.step_sim(1)
    finally:
        env.disconnect()


def parse_args():
    # Define command-line options for previewing or collecting the dataset.
    parser = argparse.ArgumentParser(description="UR5 SmolVLA PyBullet tools")
    parser.add_argument("--preview", action="store_true", help="open a single-view GUI preview")
    parser.add_argument("--preview-seconds", type=float, default=20.0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--repo-id", default="local/ur5_pick_red_cube_3cam")
    parser.add_argument("--root", default="./lerobot_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gui", action="store_true", help="collect data in DIRECT mode")
    return parser.parse_args()


def main():
    # Parse CLI arguments and choose preview mode or data collection mode.
    args = parse_args()

    # Preview mode only opens the simulator; otherwise collect demonstrations.
    if args.preview:
        preview_environment(seed=args.seed, seconds=args.preview_seconds)
        return

    collect_dataset_lerobot(
        repo_id=args.repo_id,
        root=args.root,
        num_episodes=args.episodes,
        gui=not args.no_gui,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
