import torch
import numpy as np
import gymnasium as gym
import torch.optim as optim
from collections import deque

from agent import ActorCritic, ReplayBuffer
from utils import safe_logits_to_probs

# Hyperparameters
LEARNING_RATE = 0.0001
GAMMA = 0.99
N_EPISODES = 10000
HIDDEN_DIM = 128
REPLAY_BUFFER_SIZE = 16384
BATCH_SIZE = 128
UPDATE_FREQUENCY = 1
MAX_GRAD_NORM = 0.5
SOLVED_THRESHOLD = 1000
SOLVED_EPISODE_WINDOW = 20

def train_and_play():
    # Environment
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize network, optimizer, and replay buffer
    model = ActorCritic(input_dim, n_actions, HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    total_steps = 0
    solved = False
    recent_rewards = deque(maxlen=SOLVED_EPISODE_WINDOW)

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_logits, _ = model(state_tensor)
            action_probs = safe_logits_to_probs(action_logits)
            
            try:
                if not solved:
                    action = torch.multinomial(action_probs, 1).item()
                else:
                    action = torch.argmax(action_probs).item()
            except RuntimeError:
                print(f"Error in action selection. Probs: {action_probs}")
                action = env.action_space.sample()  # Fallback to random action

            next_state, reward, done, truncated, _ = env.step(action)

            if not solved:
                replay_buffer.push(state, action, reward, next_state, done or truncated)

            state = next_state
            episode_reward += reward
            total_steps += 1

            if not solved and len(replay_buffer) > BATCH_SIZE and total_steps % UPDATE_FREQUENCY == 0:
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                # Compute TD error
                action_logits, state_values = model(states)
                _, next_state_values = model(next_states)
                td_targets = rewards + GAMMA * next_state_values.squeeze() * (1 - dones)
                td_errors = td_targets - state_values.squeeze()

                # Compute losses
                action_probs = safe_logits_to_probs(action_logits)
                actor_loss = -torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze() * td_errors.detach()
                critic_loss = td_errors.pow(2)
                loss = actor_loss.mean() + critic_loss.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        recent_rewards.append(episode_reward)

        if not solved:
            print(f"Episode {episode}, Reward: {episode_reward}, Mean recent rewards: {np.mean(recent_rewards)}")

        if not solved and len(recent_rewards) == SOLVED_EPISODE_WINDOW and np.mean(recent_rewards) >= SOLVED_THRESHOLD:
            print(f"Solved in {episode} episodes!")
            solved = True
            env.close()
            env = gym.make('CartPole-v1', render_mode='human')  # Recreate environment with rendering

        if solved and episode % 1 == 0:
            print(f"Post-solving episode {episode}, Reward: {episode_reward}")

    env.close()

if __name__ == "__main__":
    train_and_play()