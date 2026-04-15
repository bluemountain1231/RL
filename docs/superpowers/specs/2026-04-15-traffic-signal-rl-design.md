# Traffic Signal RL Thesis Project Design

## Overview
This project will build a Python + SUMO reinforcement learning system for traffic signal control, targeting a graduation thesis deliverable. The system will support a single-intersection traffic scenario, a fixed-time baseline controller, and two RL agents: DQN and DDPG. The goal is to produce a codebase that is runnable, testable, experimentally comparable, and directly mappable to thesis chapters.

## Scope
### In scope
- Single-intersection SUMO scenario
- Fixed-time baseline controller
- Gym-like SUMO environment with `reset()` and `step()`
- Separate state, action, and reward modules
- DQN agent
- DDPG agent
- Unified training entrypoint
- Unified evaluation entrypoint
- YAML configuration files
- Basic tests for state/action/reward/environment smoke path
- Standardized experiment outputs: checkpoints, CSV metrics, plots

### Out of scope
- Multi-intersection coordination
- Multi-agent reinforcement learning
- PPO, TD3, SAC, or other extra algorithms
- Prioritized replay, graph networks, attention, or advanced model tricks
- Heavyweight visualization frontends

## Architecture
The project will be split into focused modules so that environment logic, algorithm logic, training orchestration, and experiment reporting remain independent and explainable.

### Project structure
- `configs/`
  - Shared and algorithm-specific YAML configuration files
- `scenarios/`
  - SUMO network, route, and simulation config files for the single intersection
- `envs/`
  - SUMO environment orchestration, state construction, action mapping, reward calculation
- `agents/`
  - DQN, DDPG, replay buffer, and shared neural network definitions
- `training/`
  - Unified training and evaluation entrypoints
- `utils/`
  - Logging, metrics aggregation, plotting, and seeding helpers
- `tests/`
  - Basic tests for state/action/reward/environment logic
- `results/`
  - Checkpoints, logs, CSV metrics, and plots

## Environment design
### `envs/sumo_env.py`
Provides the main environment class, such as `TrafficSignalEnv`, and exposes `reset()` and `step()`. This file will orchestrate the simulation loop but will not contain detailed state, action, or reward logic.

Responsibilities:
- Start and reset SUMO/TraCI
- Advance simulation time
- Request the current observation from `state_builder.py`
- Apply validated control actions via `action.py`
- Compute reward and episode info via `reward.py`

### `envs/state_builder.py`
Constructs the RL state vector from lane- and phase-level information.

Initial state features:
- Queue length per incoming direction/lane group
- Average speed per incoming direction/lane group
- Waiting time per incoming direction/lane group
- Current signal phase one-hot encoding
- Current phase elapsed or remaining time

Outputs:
- `np.ndarray` state vector for training
- Optional detailed dictionary for debugging/analysis

### `envs/action.py`
Maps agent outputs into safe, legal traffic signal timing adjustments.

Action semantics:
- DQN outputs a discrete action index corresponding to `[-5, 0, +5]` seconds of green-time adjustment
- DDPG outputs a continuous scalar in `[-1, 1]`, which is mapped to `[-5, +5]` seconds

Constraints applied here:
- Minimum green time
- Maximum green time
- Fixed yellow duration
- Fixed all-red duration
- No illegal phase transitions

### `envs/reward.py`
Computes the reward using a weighted combination of traffic efficiency and fairness terms.

Initial reward form:
`reward = -(w1 * total_waiting + w2 * total_queue) + w3 * throughput - w4 * fairness_penalty`

Definitions:
- `total_waiting`: aggregate waiting time across all directions
- `total_queue`: aggregate queue length across all directions
- `throughput`: number of vehicles passing the intersection during the step window
- `fairness_penalty`: variance or range of directional waiting times

Design choice:
- Waiting time and queue length are dominant terms in the first version
- Throughput is a smaller positive reward
- Fairness is included with a smaller weight for thesis analysis

## Agent design
### `agents/base_agent.py`
Defines the common interface used by training code:
- `act(state, eval_mode=False)`
- `remember(transition)`
- `update()`
- `save(path)`
- `load(path)`

### `agents/dqn.py`
Implements a standard DQN agent with:
- Online Q-network
- Target Q-network
- Epsilon-greedy action selection
- Experience replay updates

First version features:
- Standard DQN only
- No Double DQN or Dueling DQN yet

### `agents/ddpg.py`
Implements a standard DDPG agent with:
- Actor network
- Critic network
- Target actor and target critic
- Soft target updates
- Replay-buffer-based updates

First version features:
- Single continuous action output
- No TD3/SAC extensions

### `agents/replay_buffer.py`
Stores transitions:
- `state`
- `action`
- `reward`
- `next_state`
- `done`

The same buffer structure will serve both DQN and DDPG.

### `agents/networks.py`
Holds shared MLP definitions.

Initial architecture:
- Two hidden layers
- Hidden width 128 or 256
- DQN head outputs Q-values for discrete actions
- DDPG actor outputs a bounded continuous action
- DDPG critic consumes concatenated state and action

## Training and evaluation design
### `training/train.py`
Unified training entrypoint for baseline-compatible experiment runs.

Responsibilities:
- Load configuration files
- Initialize environment and selected agent
- Run episode loops
- Store transitions and update agent parameters
- Record training metrics
- Periodically save checkpoints

Tracked training metrics:
- Episode reward
- Average waiting time
- Average queue length
- Throughput
- Average speed

### `training/evaluate.py`
Unified evaluation entrypoint for fair comparison.

Responsibilities:
- Load saved model checkpoints
- Disable exploration noise/random exploration
- Run fixed evaluation episodes with fixed seeds and scenario settings
- Output mean and standard deviation metrics

Comparison targets:
- Fixed-time baseline
- DQN
- DDPG

Comparison rules:
- Same SUMO scenario
- Same state features
- Same reward function
- Same seed set
- Same evaluation horizon
- Same action meaning: bounded green-time adjustment

## Configuration design
Planned config files:
- `configs/base.yaml`
- `configs/dqn.yaml`
- `configs/ddpg.yaml`
- `configs/scenario_single_intersection.yaml`
- `configs/reward.yaml`

Purpose:
- Separate experiment settings from code
- Support reproducibility and parameter studies
- Make thesis appendix tables easier to prepare

## Results and reporting
### `results/` layout
- `results/checkpoints/`
- `results/logs/`
- `results/csv/`
- `results/plots/`

Planned outputs:
- Training reward curves
- Waiting-time comparison plots
- Queue-length comparison plots
- Throughput comparison plots
- DQN vs DDPG distribution plots such as box plots

## Tests
Planned minimal test suite:
- `tests/test_state_builder.py`
- `tests/test_action_mapping.py`
- `tests/test_reward.py`
- `tests/test_env_smoke.py`

Test goals:
- Verify state vector dimension and ordering
- Verify action clipping and legal mapping
- Verify reward component composition
- Verify environment `reset()` / `step()` can complete a short smoke path

## Thesis mapping
This codebase is intentionally aligned to thesis chapter structure.

- RL theory and algorithm comparison
  - `agents/dqn.py`
  - `agents/ddpg.py`
- Traffic signal control problem modeling
  - `envs/state_builder.py`
  - `envs/action.py`
  - `envs/reward.py`
- System design and implementation
  - `scenarios/`
  - `envs/sumo_env.py`
  - `training/`
  - `configs/`
- Experiments and analysis
  - `training/evaluate.py`
  - `utils/metrics.py`
  - `utils/plotting.py`
  - `results/`

## Recommended implementation order
1. Build the SUMO single-intersection scenario and fixed-time baseline
2. Implement `TrafficSignalEnv` and verify `reset()` / `step()`
3. Implement state, action, and reward modules
4. Add the basic tests
5. Implement DQN and run the first end-to-end training loop
6. Implement DDPG under the same environment and metrics
7. Finalize unified evaluation and plotting
8. Export experimental outputs for thesis writing

## Key design decisions
- Use a mixed implementation strategy: self-built environment/experiment framework plus algorithm patterns inspired by public GitHub implementations
- Keep environment semantics under local control so the thesis can clearly justify state, action, and reward choices
- Limit the first version to one intersection so the system remains achievable and experimentally clean
- Compare baseline, DQN, and DDPG under one consistent protocol rather than building separate incompatible code paths

## Approval state
This document reflects the validated design direction agreed in brainstorming:
- Approach B mixed implementation strategy
- Thesis-standard single-intersection deliverable
- Baseline + DQN + DDPG
- Unified experiment and reporting pipeline
