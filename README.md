# CartPole-ActorCritic-RL

This project implements an Actor-Critic Reinforcement Learning algorithm to solve the CartPole problem. It demonstrates how to balance a pole on a moving cart using a deep learning approach.

## Project Description

The CartPole problem is a classic control task in reinforcement learning. The goal is to balance a pole on a cart that moves left or right. This implementation uses an Actor-Critic architecture, which combines policy-based and value-based methods to learn an optimal policy.

Key features:
- Actor-Critic neural network architecture
- Experience replay for improved sample efficiency
- Separate training and evaluation phases

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/CartPole-ActorCritic-RL.git
   cd CartPole-ActorCritic-RL
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train and evaluate the model, run:

```
python main.py
```

The script will train the agent until it solves the environment (average reward of 500 over 100 consecutive episodes) and then switch to evaluation mode with rendered visualization.

## Project Structure

- `main.py`: Main script for training and evaluation
- `agent/actor_critic.py`: Implementation of the Actor-Critic neural network
- `agent/replay_buffer.py`: Experience replay buffer implementation
- `utils/helpers.py`: Utility functions

## Hyperparameters

Key hyperparameters used in this implementation:

- Learning rate: 0.0001
- Discount factor (gamma): 0.99
- Hidden layer size: 128
- Replay buffer size: 16384
- Batch size: 128
- Update frequency: Every 1 steps

## Results

The agent typically solves the environment within a few hundred episodes. After solving, it consistently achieves the maximum episode length of 500 steps.

## Future Improvements

- Implement a target network for more stable learning
- Add support for continuous action spaces
- Experiment with different network architectures

## Contributing

Contributions to this project are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.