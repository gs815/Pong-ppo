import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pong_env import PongEnv

# -----------------------------
# Vectorized environment required by SB3
# -----------------------------
env = DummyVecEnv([lambda: PongEnv()])

# -----------------------------
# Parameters and model path
# -----------------------------
MODEL_PATH = "ppo_pong"        # will be saved as ppo_pong.zip
total_steps = 500_000          # timesteps to perform in THIS run (edit freely; see README —
                                # running this script again adds more on top, it never resets)

# -----------------------------
# Policy network size (used only if we create a new model)
# -----------------------------
policy_kwargs = dict(net_arch=[256, 256])

# -----------------------------
# Load existing model if present, otherwise create a new one
# -----------------------------
model_file_exists = os.path.exists(MODEL_PATH) or os.path.exists(MODEL_PATH + ".zip")

def new_model():
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
    )

if model_file_exists:
    try:
        print(f"Found existing model '{MODEL_PATH}'. Loading and continuing training...")
        model = PPO.load(MODEL_PATH, env=env)
    except Exception as e:
        print(f"Could not load existing model ({e}). Creating a new one instead...")
        model = new_model()
else:
    print("No existing model found. Creating a new one...")
    model = new_model()

# -----------------------------
# Training (with Ctrl+C handling so you never lose progress)
# -----------------------------
try:
    print(f"Training for {total_steps:,} timesteps... (press Ctrl+C anytime to stop and save)")
    model.learn(total_timesteps=total_steps, reset_num_timesteps=not model_file_exists)
except KeyboardInterrupt:
    print("\nTraining interrupted by user (Ctrl+C). Saving current model...")
    model.save(MODEL_PATH)
    print(f"Model saved as '{MODEL_PATH}.zip'. Run this script again later to resume training.")
    raise SystemExit

# -----------------------------
# Save trained model
# -----------------------------
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}.zip ({total_steps:,} timesteps this run, "
      f"{model.num_timesteps:,} total).")
