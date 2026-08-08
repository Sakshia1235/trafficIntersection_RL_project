
# Commanded Intersection RL Agent

A reinforcement learning agent trained to navigate a traffic intersection based on user-given direction commands (left, straight, or right).

**What is Stable Baselines3?**
Stable Baselines3 (SB3) is a Python library that provides reliable, well-tested implementations of popular reinforcement learning algorithms. This project uses PPO (Proximal Policy Optimization) — an algorithm that trains the agent by having it repeatedly attempt the task, then updating its decision-making based on what worked and what didn't. SB3 handles all the complex training machinery so the focus can stay on the environment and reward design.

**What is highway-env?**
highway-env is a collection of miniature driving environments built for autonomous vehicle research. It simulates simplified road scenarios — highways, intersections, roundabouts, parking lots — with multiple vehicles following traffic rules. Each vehicle is represented by its position, speed, and heading. This project uses the `intersection-v1` environment, where the ego car must navigate a 4-way intersection while other cars cross from different directions.

**How it works**
The agent receives the direction command (left / straight / right) as part of its observation alongside the positions and speeds of surrounding vehicles. It is rewarded for moving in the commanded direction, penalised for crashing, and cut off early via a `break_var` mechanism if it ignores the command for too long.

**How to run**
```bash
pip install highway-env stable-baselines3 gymnasium pygame numpy==1.26.4
python intersection_agent.py
```

Then enter a direction when prompted: `left`, `straight`, or `right`. The agent will train and then open a window showing the car navigating the intersection.

**Built with**
- Python
- Stable Baselines3 (PPO)
- highway-env (intersection-v1)
- Gymnasium
- PyTorch

