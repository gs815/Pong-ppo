import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

WIDTH, HEIGHT = 400, 300
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 60
BALL_SIZE = 10
PADDLE_SPEED = 6
BALL_SPEED_X = 4
BALL_SPEED_Y = 4
CPU_SPEED = 4            # max speed of the scripted CPU opponent paddle
WINNING_SCORE = 10       # points needed to end an episode (raised from 5: a lopsided
                          # 0-5 match against an early/undertrained agent felt too short)

# --- Reward tuning (tweak these values to experiment) ---
REWARD_POINT_WON = 5.0       # reward when the ball passes the CPU paddle
REWARD_POINT_LOST = -5.0     # penalty when the ball passes the agent paddle
REWARD_HIT = 1.0             # reward for hitting the ball with the paddle
REWARD_APPROACH = 0.0005     # small shaping: reward for reducing vertical distance to the ball
                              # (kept deliberately tiny: over a long episode this must not outweigh
                              # REWARD_POINT_WON/LOST and REWARD_HIT, or the agent learns to "hover"
                              # near the ball instead of actually trying to win points)
STEP_PENALTY = -0.001        # tiny penalty per step (discourages doing nothing useful)


class PongEnv(gym.Env):
    """
    Single-agent Pong environment.
    The agent controls the RIGHT paddle. The LEFT paddle is controlled by a
    simple scripted CPU (tracks the ball's y position with a fixed speed).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, max_steps=3000, agent_label="AGENT", opponent_label="CPU"):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.agent_label = agent_label        # shown above the right paddle (e.g. "AI" or "YOU")
        self.opponent_label = opponent_label  # shown above the left paddle

        # Actions: 0 = stay, 1 = up, 2 = down
        self.action_space = spaces.Discrete(3)

        # Observation: [agent_y, cpu_y, ball_x, ball_y, ball_vx, ball_vy] normalized in [-1, 1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self.screen = None
        self.clock = None
        self.font = None
        self.label_font = None

        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Pong RL")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)
            self.label_font = pygame.font.SysFont(None, 20)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_y = HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.cpu_y = HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.agent_score = 0
        self.cpu_score = 0
        self.steps = 0
        self._reset_ball(direction=self.np_random.choice([-1, 1]))
        return self._get_obs(), {}

    def _reset_ball(self, direction):
        self.ball_x = WIDTH / 2
        self.ball_y = HEIGHT / 2
        self.ball_vx = BALL_SPEED_X * direction
        self.ball_vy = BALL_SPEED_Y * self.np_random.choice([-1, 1])

    def _get_obs(self):
        return np.array([
            (self.agent_y - HEIGHT / 2) / (HEIGHT / 2),
            (self.cpu_y - HEIGHT / 2) / (HEIGHT / 2),
            (self.ball_x - WIDTH / 2) / (WIDTH / 2),
            (self.ball_y - HEIGHT / 2) / (HEIGHT / 2),
            self.ball_vx / BALL_SPEED_X,
            self.ball_vy / BALL_SPEED_Y,
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = STEP_PENALTY
        terminated = False
        truncated = False

        # --- agent paddle movement (right side) ---
        prev_dist = abs(self.agent_y + PADDLE_HEIGHT / 2 - self.ball_y)
        if action == 1:
            self.agent_y -= PADDLE_SPEED
        elif action == 2:
            self.agent_y += PADDLE_SPEED
        self.agent_y = float(np.clip(self.agent_y, 0, HEIGHT - PADDLE_HEIGHT))
        new_dist = abs(self.agent_y + PADDLE_HEIGHT / 2 - self.ball_y)

        # shaping: small reward for reducing vertical distance to the ball
        reward += REWARD_APPROACH * (prev_dist - new_dist)

        # --- CPU paddle movement (left side), simple tracking AI ---
        cpu_center = self.cpu_y + PADDLE_HEIGHT / 2
        if cpu_center < self.ball_y - 5:
            self.cpu_y += CPU_SPEED
        elif cpu_center > self.ball_y + 5:
            self.cpu_y -= CPU_SPEED
        self.cpu_y = float(np.clip(self.cpu_y, 0, HEIGHT - PADDLE_HEIGHT))

        # --- ball movement ---
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # bounce off top/bottom walls
        if self.ball_y <= 0 or self.ball_y >= HEIGHT - BALL_SIZE:
            self.ball_y = float(np.clip(self.ball_y, 0, HEIGHT - BALL_SIZE))
            self.ball_vy *= -1

        agent_x = WIDTH - PADDLE_WIDTH - BALL_SIZE   # left edge of agent paddle hit zone
        cpu_x = PADDLE_WIDTH                          # right edge of CPU paddle

        # collision with agent paddle (right) — full bounding-box overlap, not just the ball's top corner
        if self.ball_vx > 0 and self.ball_x >= agent_x and \
                self.ball_y + BALL_SIZE >= self.agent_y and self.ball_y <= self.agent_y + PADDLE_HEIGHT:
            self.ball_x = agent_x
            self.ball_vx *= -1
            reward += REWARD_HIT

        # collision with CPU paddle (left) — full bounding-box overlap, not just the ball's top corner
        if self.ball_vx < 0 and self.ball_x <= cpu_x and \
                self.ball_y + BALL_SIZE >= self.cpu_y and self.ball_y <= self.cpu_y + PADDLE_HEIGHT:
            self.ball_x = cpu_x
            self.ball_vx *= -1

        # scoring
        if self.ball_x < 0:
            self.agent_score += 1
            reward += REWARD_POINT_WON
            self._reset_ball(direction=1)
        elif self.ball_x > WIDTH:
            self.cpu_score += 1
            reward += REWARD_POINT_LOST
            self._reset_ball(direction=-1)

        if self.agent_score >= WINNING_SCORE or self.cpu_score >= WINNING_SCORE:
            terminated = True
        if self.steps >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        info = {"agent_score": self.agent_score, "cpu_score": self.cpu_score}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.screen is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.screen.fill((0, 0, 0))

        # dashed middle line
        for y in range(0, HEIGHT, 20):
            pygame.draw.rect(self.screen, (80, 80, 80), (WIDTH // 2 - 1, y, 2, 10))

        # paddles
        pygame.draw.rect(self.screen, (255, 255, 255), (0, self.cpu_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(self.screen, (255, 255, 255),
                          (WIDTH - PADDLE_WIDTH, self.agent_y, PADDLE_WIDTH, PADDLE_HEIGHT))

        # ball
        pygame.draw.rect(self.screen, (255, 255, 255), (self.ball_x, self.ball_y, BALL_SIZE, BALL_SIZE))

        # score
        score_text = self.font.render(f"{self.cpu_score}   {self.agent_score}", True, (255, 255, 255))
        self.screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 10))

        # side labels, so it's always clear which paddle is which
        left_label = self.label_font.render(self.opponent_label, True, (160, 160, 160))
        right_label = self.label_font.render(self.agent_label, True, (160, 160, 160))
        self.screen.blit(left_label, (PADDLE_WIDTH // 2 - left_label.get_width() // 2 + 10, 45))
        self.screen.blit(right_label, (WIDTH - PADDLE_WIDTH // 2 - right_label.get_width() // 2 - 10, 45))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
