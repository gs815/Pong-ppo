# Pong RL --- Reinforcement Learning Project 🏓

## 📘 Theoretical Description

Reinforcement learning is a type of machine learning in which an agent
learns to make decisions by interacting with an environment, receiving
rewards or penalties based on the actions it takes, with the goal of
maximizing the total reward over time. Through trial and error,
exploration, and progressive updates to its strategy (policy), the
agent builds increasingly effective behavior.

------------------------------------------------------------------------

## 🎯 Project Goal

Train a PPO (Proximal Policy Optimization) agent capable of playing
Pong using:

-   A custom Gymnasium-compatible environment
-   Stable-Baselines3
-   A simple scripted CPU opponent
-   Configurable reward shaping
-   Cumulative training (resume training)
-   Rendering with pygame
-   A play script where you choose to **watch the AI play**, or
    **play yourself** against the CPU

------------------------------------------------------------------------

## 📂 Project Structure

    Pong-ppo-main/
    ├─ pong_env.py        # Custom Gymnasium environment
    ├─ train_pong.py      # PPO training script
    ├─ play_pong.py       # AI vs CPU, or You vs CPU
    └─ ppo_pong.zip       # Saved model after training

------------------------------------------------------------------------

## ⚙️ Requirements

### 1. Install Python 3.11

This project is tested on **Python 3.11**. Check if you already have it:

``` bash
py -3.11 --version
```

(on macOS/Linux try `python3.11 --version`)

If that prints `Python 3.11.x`, skip to step 2. Otherwise, install it:

**Windows**

-   Easiest: open a terminal and run `py install 3.11.9` — downloads and
    installs it for you, no browser needed.
-   Or download it manually: [python-3.11.9-amd64.exe](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)
    (this is the last 3.11 release with an official Windows installer —
    newer 3.11.x patches are security-only, source-code releases).
    Run it and make sure **"Add python.exe to PATH"** is checked before
    clicking Install, or the `python`/`py` commands won't work afterwards.

**macOS**

-   With [Homebrew](https://brew.sh): `brew install python@3.11`
-   Or download the installer from [python.org/downloads/release/python-3119](https://www.python.org/downloads/release/python-3119/) (macOS 64-bit universal2 installer)

**Linux**

-   Debian/Ubuntu: `sudo apt install python3.11 python3.11-venv`
-   Fedora: `sudo dnf install python3.11`
-   Or build from the source tarball linked on the same release page, if your distro doesn't package it

### 2. Install the dependencies

From inside the project folder, run:

``` bash
py -3.11 -m pip install gymnasium stable-baselines3 torch pygame numpy
```

(on macOS/Linux: `python3.11 -m pip install gymnasium stable-baselines3 torch pygame numpy`)

> 💡 **Note on CPU vs GPU:** when training starts you'll see `Using cpu
> device` printed — this is expected and fine. The network here is tiny
> (256×256 neurons on a 6-number input), so CPU is actually fast enough
> and often faster than shuffling such a small workload to a GPU. This
> also means an AMD GPU is not a limitation here: PyTorch's GPU
> acceleration (CUDA) is NVIDIA-only, but you don't need it for this
> project either way.

------------------------------------------------------------------------

## 🚀 How to Use

### 1️⃣ Training

``` bash
py -3.11 train_pong.py
```

-   If `ppo_pong.zip` exists → continues training
-   If it does not exist → creates a new model

**Training is cumulative across runs — it never resets.** Each run trains
for `total_steps` (500,000 by default — edit this number in `train_pong.py`
if you want shorter or longer runs) and adds that on top of whatever the
model already learned. So you can split training into several separate
sessions instead of one long sitting: run the script, stop, check progress
with `play_pong.py`, then run it again later — the agent keeps improving
from where it left off rather than starting over. The same applies if you
press Ctrl+C mid-run: progress up to that point is saved before exiting.

### 2️⃣ Watch it play / play yourself

``` bash
py -3.11 play_pong.py
```

You'll be asked to choose:

-   `1` → **AI vs CPU** (loads `ppo_pong.zip` and watches it play)
-   `2` → **You vs CPU** (control the right paddle with ↑/↓ or W/S)

------------------------------------------------------------------------

## 🧠 Reward Shaping

Configurable in `pong_env.py`:

``` python
REWARD_POINT_WON = 5.0
REWARD_POINT_LOST = -5.0
REWARD_HIT = 1.0
REWARD_APPROACH = 0.0005
STEP_PENALTY = -0.001
```

Reward components:

-   + Large reward for scoring a point against the CPU
-   − Large penalty for conceding a point
-   + Reward for hitting the ball with the paddle
-   + Small reward for reducing vertical distance to the ball (tracking)
-   − Tiny penalty per step

------------------------------------------------------------------------

## ⏱ Match Length

Configurable in `pong_env.py`:

``` python
WINNING_SCORE = 10   # points needed to win a match
max_steps = 3000     # default safety cap (PongEnv.__init__ parameter), ~50 seconds at 60 FPS
```

A match ends when either side reaches `WINNING_SCORE`, or after `max_steps`
if no one does (this happens often between two well-matched players, since
rallies can go on for a long time). Each side's name (`CPU`, `AI`, or `YOU`)
is shown above its paddle during play.

------------------------------------------------------------------------

## 🏗 Architecture

-   Algorithm: PPO
-   Policy: MLP (2 layers of 256 neurons)
-   Agent controls the **right** paddle; a scripted AI controls the **left** (CPU) paddle
-   Observation: 6 normalized values (both paddle y positions, ball x/y, ball x/y velocity)
-   Cumulative training supported

------------------------------------------------------------------------

## 📌 Final Note

This project is designed as a complete educational exercise in
Reinforcement Learning: from building the environment to visualizing the
trained agent — and challenging it yourself.
