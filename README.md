
# Commanded Intersection RL Agent

The aim of this project is to build an autonomous vehicle agent that can navigate a traffic intersection by following real-time direction commands from a user, while safely handling other vehicles in the environment.

**What is Stable Baselines3?**
Stable Baselines3 (SB3) is a Python library that provides reliable, well-tested implementations of popular reinforcement learning algorithms. This project uses PPO (Proximal Policy Optimization) — an algorithm that trains the agent by having it repeatedly attempt the task, then updating its decision-making based on what worked and what didn't. SB3 handles all the complex training machinery so the focus can stay on the environment and reward design.

**What is highway-env?**
highway-env is a collection of miniature driving environments built for autonomous vehicle research. It simulates simplified road scenarios — highways, intersections, roundabouts, parking lots — with multiple vehicles following traffic rules. Each vehicle is represented by its position, speed, and heading. This project uses the `intersection-v1` environment, where the ego car must navigate a 4-way intersection while other cars cross from different directions.

**How it works**
The agent receives the direction command (left / straight / right) as part of its observation alongside the positions and speeds of surrounding vehicles. It is rewarded for moving in the commanded direction, penalised for crashing, and cut off early via a `break_var` mechanism if it ignores the command for too long.

**Performance**

The agent was trained for 10,000 timesteps on CPU using PPO. Performance was measured by running 3 demo episodes with a length of 10 steps after training and recording the total reward per episode and whether the agent crashed.

Results after training:

Episode 1: reward = 0.55 — OK
Episode 2: reward = 9.47 — OK
Episode 3: reward = 3.60 — OK

2 out of 3 episodes consistently achieved positive rewards with no crashes, showing the agent learned basic collision avoidance and intersection navigation. Reward was used as the primary performance metric higher reward means the agent survived longer, moved at a reasonable speed, and followed the commanded direction more accurately

**How to run**
```bash
pip install highway-env stable-baselines3 gymnasium pygame numpy==1.26.4
python project_autonomous_car.py
```

Then enter a direction when prompted: `left`, `straight`, or `right`. The agent will train and then open a window showing the car navigating the intersection.

**Built with**
- Python
- Stable Baselines3 (PPO)
- highway-env (intersection-v1)
- Gymnasium
- PyTorch

