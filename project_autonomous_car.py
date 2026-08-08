import gymnasium as gym
import highway_env
import numpy as np
from stable_baselines3 import PPO
import torch
import time

# ── direction setup ────────────────────────────────────────────────────────────

DIRECTION_MAP = {'left': 0, 'straight': 1, 'right': 2}

# Expected heading (radians) after clearing intersection.
# intersection-v0 ego approaches from South → heading North (~π/2)
EXPECTED_HEADING = {
    'left':     0.0,
    'straight': np.pi / 2,
    'right':    np.pi,
}

MAX_HEADING_ERROR = np.pi / 3   # 60° tolerance before counting as wrong
WRONG_DIR_LIMIT   = 30          # steps going wrong direction before break_var fires


# ── environment ────────────────────────────────────────────────────────────────

class CommandedIntersectionEnv(gym.Wrapper):
    def __init__(self, command='straight', render_mode=None):
        assert command in DIRECTION_MAP
        env = gym.make('intersection-v0', render_mode=render_mode)
        super().__init__(env)

        self.command         = command
        self.command_encoded = DIRECTION_MAP[command]
        self._prev_action    = None
        self._wrong_dir_steps = 0

        obs_shape = self.observation_space.shape   # (15, 7)
        flat_size = obs_shape[0] * obs_shape[1] + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_size,), dtype=np.float32
        )

    def _augment(self, obs):
        flat = obs.flatten().astype(np.float32)
        cmd  = np.array([self.command_encoded / 2.0], dtype=np.float32)
        return np.concatenate([flat, cmd])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action     = None
        self._wrong_dir_steps = 0
        return self._augment(obs), info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # ── break_var logic ───────────────────────────────────────────────────
        #   Check if the car is heading the wrong way while actually moving.
        #   If it misbehaves for WRONG_DIR_LIMIT consecutive steps → break_var
        #   fires by forcing truncated=True so the episode ends immediately.
        ego     = obs[0]
        vx, vy  = float(ego[3]), float(ego[4])
        speed   = np.hypot(vx, vy)

        break_var = False
        if speed > 1.0:
            heading       = np.arctan2(float(ego[6]), float(ego[5]))
            target        = EXPECTED_HEADING[self.command]
            heading_error = abs(_angle_diff(heading, target))

            if heading_error > MAX_HEADING_ERROR:
                self._wrong_dir_steps += 1
            else:
                self._wrong_dir_steps = 0   # reset if car corrects itself

            if self._wrong_dir_steps >= WRONG_DIR_LIMIT:
                break_var = True
                truncated = True             # end episode now
                print(f"  ⚠  break_var fired — agent ignored '{self.command}' command")

        reward = self._reward(obs, action, base_reward, info, break_var)
        self._prev_action = action
        return self._augment(obs), reward, terminated, truncated, info

    def _reward(self, obs, action, base_reward, info, break_var):
        ego     = obs[0]
        speed   = np.hypot(float(ego[3]), float(ego[4]))
        heading = np.arctan2(float(ego[6]), float(ego[5]))

        r = base_reward                     # env's arrival bonus lives here
        r += 0.05                           # survival bonus

        if info.get('crashed', False):
            r -= 10.0

        if break_var:
            r -= 5.0                        # extra penalty for ignoring command

        # direction reward — only when moving
        if speed > 1.0:
            target = EXPECTED_HEADING[self.command]
            error  = abs(_angle_diff(heading, target))
            r += 0.3 * np.cos(error)        # +0.3 perfect, 0 at 90°, -0.3 opposite

        # speed reward — gaussian centred at 7 m/s
        r += 0.05 * np.exp(-0.5 * ((speed - 7.0) / 4.0) ** 2)

        # smoothness
        if self._prev_action is not None and action != self._prev_action:
            r -= 0.05

        return r


# ── utility ────────────────────────────────────────────────────────────────────

def _angle_diff(a, b):
    """Shortest signed angular difference, result in (-π, π]."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi

def _get_command():
    while True:
        raw = input("Enter direction (left / straight / right): ").strip().lower()
        if raw in DIRECTION_MAP:
            return raw
        print(f"  Invalid — choose from: {', '.join(DIRECTION_MAP)}")



# ── main: train then run ───────────────────────────────────────────────────────

def train_and_run(command=None, timesteps=10000, n_demo_episodes=3):
    if command is None:
        command = _get_command()

    
    # 1. TRAIN
    train_env = CommandedIntersectionEnv(command=command)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
    )
    model.learn(total_timesteps=timesteps)
    model.save(f"intersection_ppo_{command}")
    train_env.close()
    print("Training done.\n")

    # 2. RUN (render on)
    print(f"── Demo | command='{command}' | {n_demo_episodes} episode(s) ──")
    for ep in range(1, n_demo_episodes + 1):
        demo_env  = CommandedIntersectionEnv(command=command, render_mode='human')
        obs, info = demo_env.reset()
        done = truncated = False
        ep_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = demo_env.step(action)
            ep_reward += reward
            time.sleep(0.05)

        crashed = info.get('crashed', False)
        print(f"  Episode {ep}: reward={ep_reward:.2f}  {'CRASHED' if crashed else 'OK'}")
        input("Press Enter for next episode...")   # ← window stays open
        demo_env.close()


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_and_run()