import os
import pygame
from stable_baselines3 import PPO
from pong_env import PongEnv

MODEL_PATH = "ppo_pong"


def choose_mode():
    print("\nChoose a mode:")
    print("  1) AI vs CPU   -> watch the trained agent play")
    print("  2) You vs CPU  -> play yourself (Up/Down arrows or W/S)")
    choice = input("Mode (1/2): ").strip()
    return "human" if choice == "2" else "ai"


def play(mode):
    if mode == "ai" and not (os.path.exists(MODEL_PATH) or os.path.exists(MODEL_PATH + ".zip")):
        print(f"\n⚠️  No model found ('{MODEL_PATH}.zip' does not exist in this folder).")
        print("   Run 'py -3.11 train_pong.py' first to train the agent,")
        print("   or choose mode 2 (You vs CPU) to play right away.")
        return

    agent_label = "AI" if mode == "ai" else "YOU"
    env = PongEnv(render_mode="human", agent_label=agent_label)
    obs, info = env.reset()

    model = None
    if mode == "ai":
        model = PPO.load(MODEL_PATH)
        print(f"\nModel loaded (trained so far on {model.num_timesteps:,} timesteps).")
        print("AI vs CPU in progress... close the window to stop.")
    else:
        print("\nYou vs CPU in progress... use Up/Down arrows (or W/S). Close the window to stop.")

    done = False
    while not done:
        if mode == "ai":
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = 0  # stay
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                action = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                action = 2

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    player_label = "AI" if mode == "ai" else "You"
    if info["cpu_score"] > info["agent_score"]:
        winner = "CPU"
    elif info["agent_score"] > info["cpu_score"]:
        winner = player_label
    else:
        winner = "Draw"
    print(f"\nMatch over! Score -> CPU: {info['cpu_score']}  "
          f"{player_label}: {info['agent_score']}  |  Winner: {winner}")

    env.close()


if __name__ == "__main__":
    selected_mode = choose_mode()
    play(selected_mode)
