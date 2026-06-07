import csv
import time

import numpy as np
import pybullet as p
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame

from ur5_smolvla_env import UR5PickCubeEnv


# =========================================================
# Config
# =========================================================
# Path to the trained policy checkpoint directory.
POLICY_PATH = "outputs1/train/smolvla_ur5_pick_cube/checkpoints/last/pretrained_model"

# Dataset configuration used to load LeRobot metadata and feature definitions for inference.
DATASET_REPO_ID = "local/ur5_pick_cube_random_v1_3cam"
DATASET_ROOT = "./lerobot_dataset"

# Evaluation episode count and simulation runtime settings.
EPISODES_PER_TASK = 20
MAX_STEPS = 100
GUI = True
DEVICE = "cuda"  # Options: "cuda" / "cpu" / "auto"
SLEEP_SEC = 0.0

# Language task prompts to evaluate.
TEST_TASKS = [
    "pick up the object on the table",
    "pick red object",
    "pick up the cube for me",
    "lift the red block",
    "pick up the red cube with the gripper",
]

# Action safety limits to prevent large policy outputs from hitting the table or diverging.
MIN_EE_Z = 0.775
MAX_DELTA_POS = 0.02

# Keep this empty if the dataset is not a multi-embodiment dataset.
ROBOT_TYPE = ""

# Output file for evaluation results.
RESULT_CSV_PATH = "task_eval_results.csv"


# =========================================================
# Utils
# =========================================================
def get_device(device_mode="auto"):
    """Return the torch device according to the configured device mode."""
    if device_mode != "auto":
        return torch.device(device_mode)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def extract_action(action_output):
    """
    Normalize the action format returned by SmolVLA / LeRobot to shape=(7,).
    """
    # Unwrap dictionary outputs that contain the action tensor/array.
    if isinstance(action_output, dict):
        if "action" in action_output:
            action_output = action_output["action"]
        else:
            raise KeyError(f"Unknown action dict keys: {list(action_output.keys())}")

    # Move torch tensors to CPU and convert them to NumPy arrays.
    if torch.is_tensor(action_output):
        action_output = action_output.detach().cpu().numpy()

    action_output = np.asarray(action_output, dtype=np.float32)

    # Remove batch/time dimensions if the policy returns them.
    if action_output.ndim == 3:
        action_output = action_output[0, 0]
    elif action_output.ndim == 2:
        action_output = action_output[0]

    # Validate that the final action vector matches the expected UR5 action size.
    if action_output.shape[-1] != 7:
        raise ValueError(f"Expected action dim = 7, but got shape {action_output.shape}")

    return action_output.astype(np.float32)


def gripper_touching_table(env):
    """Check whether the gripper touches the table for safety debugging."""
    # Query all contacts between the robot and the table.
    contacts = p.getContactPoints(bodyA=env.robot.id, bodyB=env.table_id)
    for contact in contacts:
        if contact[3] >= env.robot.eef_id:
            return True
    return False
def gripper_touching_cube(env):
    """Check whether the gripper fingers touch the cube."""
    # Query all contacts between the robot and the cube.
    contacts = p.getContactPoints(bodyA=env.robot.id, bodyB=env.cube_id)

    for contact in contacts:
        robot_link = contact[3]

        # Robotiq left and right finger links.
        if robot_link in [12, 17]:
            return True

    return False


def show_gui_text(env, text, color=(1, 0, 0), life_time=1.5):
    """Display success or failure text in the PyBullet GUI."""
    # Skip GUI text rendering when the environment is running headless.
    if not getattr(env, "gui", False):
        return

    p.addUserDebugText(
        text=text,
        textPosition=[0.25, -0.45, 1.25],
        textColorRGB=color,
        textSize=1.8,
        lifeTime=life_time,
    )

def safe_action(env, action):
    """
    Apply safety constraints to the action predicted by the policy:
    - limit xyz displacement per step to MAX_DELTA_POS
    - clamp the gripper opening between 0.0 and 0.085
    - prevent the end effector from going below the safe table height
    """
    # Work on a copy so the original raw action can still be logged.
    action = np.asarray(action, dtype=np.float32).copy()

    action[:3] = np.clip(action[:3], -MAX_DELTA_POS, MAX_DELTA_POS)
    action[6] = np.clip(action[6], 0.0, 0.085)

    # Predict the next end-effector position and clamp it above the safety floor.
    ee_pos, _ = env.robot.get_ee_pose()
    next_pos = ee_pos + action[:3]

    if next_pos[2] < MIN_EE_Z:
        next_pos[2] = MIN_EE_Z

    action[:3] = next_pos - ee_pos
    return action


# =========================================================
# SmolVLA Policy Runner
# =========================================================
class SmolVLAPolicyRunner:
    """Load the SmolVLA policy and convert environment observations into LeRobot inference format."""

    def __init__(self, policy_path, dataset_repo_id, dataset_root, device="auto"):
        # Resolve the device first so all model and preprocessing tensors use the same target.
        self.device = get_device(device)
        self.current_task = TEST_TASKS[0]

        print("=========================================================")
        print("Loading SmolVLA policy")
        print("=========================================================")
        print("policy_path     :", policy_path)
        print("dataset_repo_id :", dataset_repo_id)
        print("dataset_root    :", dataset_root)
        print("device          :", self.device)
        print("=========================================================")

        # Load the trained policy weights.
        self.model = SmolVLAPolicy.from_pretrained(policy_path)
        self.model.to(self.device)
        self.model.eval()

        # Load dataset features required by build_inference_frame.
        dataset = LeRobotDataset(repo_id=dataset_repo_id, root=dataset_root)
        self.dataset_features = dataset.meta.features

        # Create the preprocessing and postprocessing pipelines used for LeRobot policy inference.
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            policy_path,
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)}
            },
        )

        print("[INFO] Policy and processors ready.")
        print("[INFO] Dataset feature keys:")
        for key in self.dataset_features.keys():
            print("  ", key)

    def set_task(self, task):
        """Set the current language task prompt for the VLA policy."""
        self.current_task = task

    def reset(self):
        """Reset the policy before each episode if it maintains an internal temporal state."""
        if hasattr(self.model, "reset"):
            self.model.reset()

    def build_obs_frame(self, obs):
        """Convert environment observations into a LeRobot inference frame."""
        # Read and validate the robot joint state from the environment observation.
        joint_positions = obs["joint_positions"].astype(np.float32)

        if joint_positions.shape != (7,):
            raise ValueError(
                f"Expected obs['joint_positions'] shape == (7,), got {joint_positions.shape}"
            )

        # LeRobot inference frames use flattened observation keys.
        raw_obs = {
            "camera1": obs["image_camera1"],
            "camera2": obs["image_camera2"],
            "camera3": obs["image_camera3"],
            "joint_0": float(joint_positions[0]),
            "joint_1": float(joint_positions[1]),
            "joint_2": float(joint_positions[2]),
            "joint_3": float(joint_positions[3]),
            "joint_4": float(joint_positions[4]),
            "joint_5": float(joint_positions[5]),
            "gripper": float(joint_positions[6]),
        }

        print("[DEBUG] raw_obs keys:", list(raw_obs.keys()))
        print("[DEBUG] joint_positions:", np.round(joint_positions, 4).tolist())
        print("[DEBUG] task:", self.current_task)

        # Build the final LeRobot-compatible frame consumed by the preprocessors.
        obs_frame = build_inference_frame(
            observation=raw_obs,
            ds_features=self.dataset_features,
            device=self.device,
            task=self.current_task,
            robot_type=ROBOT_TYPE,
        )
        return obs_frame

    @torch.no_grad()
    def predict(self, obs):
        """Predict a 7-dimensional action from the current environment observation."""
        # Build and preprocess the observation frame before querying the model.
        obs_frame = self.build_obs_frame(obs)
        policy_input = self.preprocess(obs_frame)

        action_out = self.model.select_action(policy_input)
        action_out = self.postprocess(action_out)

        # Convert the model output to the environment action format.
        action = extract_action(action_out)

        if not np.all(np.isfinite(action)):
            raise ValueError(f"Predicted action contains NaN/Inf: {action}")

        return action


# =========================================================
# Episode Runner
# =========================================================
def run_episode(env, policy, max_steps=100, sleep_sec=0.0, verbose=True):
    """Run a single episode and return True if the task succeeds."""
    # Reset both the environment and the policy state before the episode starts.
    obs = env.reset()
    policy.reset()
    has_started_closing = False
    for step in range(max_steps):
        # Query the policy for the next raw action.
        try:
            raw_action = policy.predict(obs)
        except Exception as e:
            print(f"[ERROR] policy.predict failed at step {step}: {e}")
            return False

        # Apply safety constraints before sending the action to the simulator.
        action = safe_action(env, raw_action)

        if verbose:
            print(
                f"[STEP {step:03d}] "
                f"raw_action={np.round(raw_action, 4).tolist()} "
                f"safe_action={np.round(action, 4).tolist()} "
                f"ee_pos={np.round(obs['ee_pos'], 4).tolist()} "
                f"cube_pos={np.round(obs['cube_pos'], 4).tolist()} "
                f"joint_pos={np.round(obs['joint_positions'], 4).tolist()}"
            )
        # Treat a small gripper command as the start of a closing motion.
        is_closing_gripper = action[6] <= 0.02

       
        # Fail early if the gripper hits the cube before starting to close.
        if not has_started_closing and not is_closing_gripper:
            if gripper_touching_cube(env):
                msg = "FAIL: gripper hit cube before closing"
                print("[FAIL]", msg)
                show_gui_text(env, msg, color=(1, 0, 0))
                time.sleep(1.0)
                return False

        if is_closing_gripper:
            has_started_closing = True
        # Step the simulator with the safe action and receive the next observation.
        obs, reward, done, info = env.apply_action(action)

        if gripper_touching_table(env) and verbose:
            print("[WARN] gripper touched table")

        if env.is_success():
            if verbose:
                print("Success")
            return True

        if done:
            if verbose:
                print("Env done")
            return False

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    return env.is_success()


# =========================================================
# Evaluation per task
# =========================================================
def evaluate_task(env, policy, task, num_episodes, max_steps, sleep_sec=0.0, verbose=False):
    """Run multiple episodes for one task and return both the summary and per-episode results."""
    # Update the policy prompt before evaluating this task.
    policy.set_task(task)

    successes = 0
    results = []

    print("\n=========================================================")
    print(f"Evaluating task: {task}")
    print(f"Episodes: {num_episodes}")
    print("=========================================================")

    # Run the requested number of episodes and track the running success rate.
    for ep in range(num_episodes):
        print(f"[TASK={task}] Episode {ep + 1}/{num_episodes}")

        success = run_episode(
            env=env,
            policy=policy,
            max_steps=max_steps,
            sleep_sec=sleep_sec,
            verbose=verbose,
        )

        successes += int(success)
        running_sr = successes / (ep + 1)

        # Store episode-level metrics for later CSV export.
        results.append(
            {
                "task": task,
                "episode": ep + 1,
                "success": int(success),
                "running_success_rate": running_sr,
            }
        )

        print(
            f"[RESULT] task='{task}' ep={ep + 1} "
            f"success={int(success)} "
            f"running_success_rate={running_sr:.4f}"
        )

    # Build the final task-level summary.
    final_sr = successes / num_episodes
    summary = {
        "task": task,
        "successes": successes,
        "episodes": num_episodes,
        "success_rate": final_sr,
    }

    return summary, results


def save_results_csv(task_summaries, episode_results, csv_path):
    """Write task summaries and episode-level results to separate CSV files."""
    summary_path = csv_path.replace(".csv", "_summary.csv")

    # Task-level summary: number of successes and success rate for each task.
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "successes", "episodes", "success_rate"],
        )
        writer.writeheader()
        for row in task_summaries:
            writer.writerow(row)

    # Episode-level details: success status and running success rate for each episode.
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "episode", "success", "running_success_rate"],
        )
        writer.writeheader()
        for row in episode_results:
            writer.writerow(row)

    print(f"[INFO] Episode-level results saved to: {csv_path}")
    print(f"[INFO] Summary results saved to: {summary_path}")


# =========================================================
# Main
# =========================================================
def main():
    """Create the environment and policy, then evaluate all tasks in TEST_TASKS sequentially."""
    # Create the simulation environment.
    env = UR5PickCubeEnv(gui=GUI)

    # Create the policy runner that handles model loading and inference formatting.
    policy = SmolVLAPolicyRunner(
        policy_path=POLICY_PATH,
        dataset_repo_id=DATASET_REPO_ID,
        dataset_root=DATASET_ROOT,
        device=DEVICE,
    )

    all_task_summaries = []
    all_episode_results = []

    try:
        # Evaluate each language task and accumulate all results.
        for task in TEST_TASKS:
            summary, episode_results = evaluate_task(
                env=env,
                policy=policy,
                task=task,
                num_episodes=EPISODES_PER_TASK,
                max_steps=MAX_STEPS,
                sleep_sec=SLEEP_SEC,
                verbose=False,  # Set to True to print step-by-step details.
            )

            all_task_summaries.append(summary)
            all_episode_results.extend(episode_results)

        print("\n=========================================================")
        print("FINAL SUMMARY")
        print("=========================================================")
        for row in all_task_summaries:
            print(
                f"task={row['task']} | "
                f"successes={row['successes']} | "
                f"episodes={row['episodes']} | "
                f"success_rate={row['success_rate']:.4f}"
            )
        print("=========================================================")

        # Save both the per-task summaries and the episode-level records.
        save_results_csv(
            task_summaries=all_task_summaries,
            episode_results=all_episode_results,
            csv_path=RESULT_CSV_PATH,
        )

    finally:
        # Always release the simulator connection, even if evaluation fails.
        env.disconnect()
        print("[INFO] Environment disconnected.")


if __name__ == "__main__":
    main()
