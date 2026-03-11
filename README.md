# Scalable Swarm Control Using Deep Reinforcement Learning

A ROS 2 package that deploys a Deep Reinforcement Learning policy on real Crazyflie drones to track a dynamic target (QCar ground vehicle) while avoiding obstacles. Agents are trained individually in a static single-agent environment using Soft Actor-Critic (SAC), then the resulting policy is scaled to multi-agent swarm scenarios — enabling collision-free target tracking using only local neighborhood information.

![Target tracking with two obstacles](/trajectory_one_vs_one_two_obst.gif)

---

## Overview

Swarm control is challenging due to the non-stationarity introduced by dynamic agent interactions and the scalability limitations of standard multi-agent RL. This project addresses both by training a single-agent policy and deploying it across multiple agents:

1. **Stage 1 — Single-Agent Training:** A Soft Actor-Critic (SAC) policy is trained in a 2D simulated environment. One agent learns to reach a target while avoiding static obstacles.
2. **Stage 2 — Multi-Agent Deployment:** The trained policy is deployed on each drone independently. Other drones are treated as dynamic obstacles. Only local neighborhood information is used, making the approach fully decentralized and scalable.

Each agent observes:
- Distance and line-of-sight angle to the target
- Its own heading angle
- Distance and line-of-sight angle to neighboring agents/obstacles

---

## Repository Structure

```
target_following-main/
├── target_tracking/
│   ├── core.py              # Environment and agent dynamics (particle & Dubins models)
│   ├── utils.py             # SAC policy loader (Tactic class) and observation utilities
│   ├── commands_node.py     # Main ROS 2 node: reads poses, runs DRL policy, publishes velocity commands
│   ├── send_commands_node.py # Relay node: republishes velocity commands at high frequency
│   └── __init__.py
├── package.xml
└── setup.py
```

---

## Requirements

- **ROS 2 Humble**
- **Python 3.10+**
- **Crazyflie drones** with motion capture tracking
- **QCar** (Quanser) as the dynamic target

Python dependencies:
```
stable-baselines3
gymnasium
numpy
scipy
icecream
torch
```

External ROS 2 packages:
- [`crazyswarm2`](https://github.com/dimitriasilveria/crazyswarm2.git)
- [`motion_capture_tracking`](https://github.com/IMRCLab/motion_capture_tracking.git)
- [`controller_pkg`](https://github.com/dimitriasilveria/controller_pkg.git)

---

## Installation

1. Clone this repository and the required packages into your ROS 2 workspace:

    ```bash
    cd ~/ros2_ws/src
    git clone <this-repo-url>
    git clone https://github.com/dimitriasilveria/crazyswarm2.git
    git clone https://github.com/IMRCLab/motion_capture_tracking.git
    git clone https://github.com/dimitriasilveria/controller_pkg.git
    ```

2. Build the workspace:

    ```bash
    cd ~/ros2_ws
    colcon build --packages-select target_tracking
    ```

3. Source the workspace (add to `.bashrc` to avoid running this every time):

    ```bash
    source /opt/ros/humble/setup.bash
    source ~/ros2_ws/install/local_setup.bash
    ```

4. Place your trained SAC model inside the `Models/` directory and update the `dir_models` path in `commands_node.py`.

---

## Running on Real Hardware

> **Safety first:** Always have the landing node ready before starting any flight.

### Step 1 — Configure Crazyflies

1. Open [crazyflies.yaml](crazyswarm2/crazyflie/config/crazyflies.yaml)
2. Under each drone's name, set `enable: true`
3. Make sure all drones are tagged and visible in the motion capture system

### Step 2 — Open a safety landing terminal

In a dedicated terminal, run the landing node. Press `Enter` at any moment to land all drones:

```bash
ros2 run controller_pkg landing
```

### Step 3 — Launch the full system

In a new terminal, launch the motion capture, watchdog, Crazyflie server, and target tracking nodes:

```bash
ros2 launch target_tracking target_tracking_launch.py
```

The `commands_node` will automatically:
- Subscribe to `/poses` from the motion capture system
- Identify the controlled drone, the target (QCar), and neighboring agents
- Build the observation vector and query the SAC policy
- Publish velocity commands to `/robot/cmd_vel_slow`

### Step 4 — Start target tracking

After all drones have taken off, start the tracking behavior:

```bash
ros2 run controller_pkg encircling
```

Press `Enter` in the encircling node terminal to send the start flag. The drones will begin tracking the QCar target.

**If anything goes wrong, switch to the landing terminal and press `Enter`.**

---

## Node Reference

### `commands_node`

Main control node. Runs at 100 Hz.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `robot` | `C20` | Name of the controlled Crazyflie |
| `n_agents` | `3` | Total number of agents (including self) |
| `hover_height` | `0.5` | Takeoff height in meters |

Subscriptions:
- `/poses` (`NamedPoseArray`) — Motion capture poses for all agents and target
- `/landing` (`Bool`) — Triggers emergency landing
- `/encircle` (`Bool`) — Starts the tracking behavior

Publishers:
- `/{robot}/cmd_vel_slow` (`Twist`) — Velocity command for the drone
- `/{robot}/cmd_vel_stamped` (`TwistStamped`) — Stamped version for logging

### `send_commands_node`

Relay node that re-publishes velocity commands at a fixed 100 Hz rate to ensure consistent command delivery to the drone firmware.

---

## Experimental Results

The policy was evaluated over R = 100 runs in simulation with different swarm sizes (N) and policy observation capacity (α_π):

| Scenario | N agents | α_π | Mean dist. to target | Collisions |
|----------|----------|-----|----------------------|------------|
| 1A | 3 | 2 | < 30 m | 0.2 |
| 2A | 4 | 3 | < 35 m | 0.8 |
| 3A | 5 | 4 | < 30 m | 7.5 |
| 4A | 6 | 5 | < 35 m | 5.6 |
| 1B (scaled) | 6 | 2 | < 30 m | 1.5 |
| 2B (scaled) | 10 | 3 | < 57 m | 8.8 |

Scenarios 1B and 2B demonstrate scalability: the policy trained observing only 2–3 agents successfully controls a swarm of 6–10 drones using only local neighborhood information.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{Silveria2025,
  author    = {Silveria, Dimitria and Cabral, Kleber and Givigi, Sidney},
  title     = {Scalable Swarm Control Using Deep Reinforcement Learning},
  booktitle = {2025 IEEE International Systems Conference (SysCon)},
  year      = {2025},
  doi       = {10.1109/SYSCON64521.2025.11014655}
}
```
