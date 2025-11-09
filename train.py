"""
Training script for Case Closed agent using Deep Q-Network (DQN) reinforcement learning.

This script trains a neural network to play the Case Closed game by:
1. Playing games against itself or a random opponent
2. Collecting experience (state, action, reward, next_state)
3. Training the model using Q-learning
4. Saving the trained model for use in agent.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os
import json

from case_closed_game import Game, Direction, GameResult


class DQN(nn.Module):
    """Deep Q-Network for learning optimal actions in Case Closed game."""
    
    def __init__(self, state_size=11, action_size=4, hidden_size=128):
        """
        Args:
            state_size: Number of features in state representation (11 for our state)
            action_size: Number of possible actions (4: LEFT, RIGHT, UP, DOWN)
            hidden_size: Size of hidden layers
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """Forward pass through the network."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def get_state(game, my_agent, player_number):
    """Get state representation (same as in agent.py)."""
    head = my_agent.trail[-1]
    board = game.board
    
    # Calculate adjacent positions (with torus wrapping)
    point_l = ((head[0] - 1) % board.width, head[1])
    point_r = ((head[0] + 1) % board.width, head[1])
    point_u = (head[0], (head[1] - 1) % board.height)
    point_d = (head[0], (head[1] + 1) % board.height)
    
    # Current direction flags
    dir_l = my_agent.direction == Direction.LEFT
    dir_r = my_agent.direction == Direction.RIGHT
    dir_u = my_agent.direction == Direction.UP
    dir_d = my_agent.direction == Direction.DOWN
    
    # Get opponent agent
    other_agent = game.agent2 if player_number == 1 else game.agent1
    
    # Check collisions (danger detection)
    def is_collision(pos):
        cell_state = board.get_cell_state(pos)
        if cell_state == 1:  # AGENT cell
            if pos in my_agent.trail and pos != head:
                return True
            if other_agent.alive and pos in other_agent.trail:
                return True
        return False
    
    # Get opponent's head position
    opp_head = other_agent.trail[-1] if other_agent.alive else head
    
    state = [
        # Danger straight
        (dir_r and is_collision(point_r)) or 
        (dir_l and is_collision(point_l)) or 
        (dir_u and is_collision(point_u)) or 
        (dir_d and is_collision(point_d)),
        
        # Danger right
        (dir_u and is_collision(point_r)) or 
        (dir_d and is_collision(point_l)) or 
        (dir_l and is_collision(point_u)) or 
        (dir_r and is_collision(point_d)),
        
        # Danger left
        (dir_d and is_collision(point_r)) or 
        (dir_u and is_collision(point_l)) or 
        (dir_r and is_collision(point_u)) or 
        (dir_l and is_collision(point_d)),
        
        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,
        
        # Opponent location
        (opp_head[0] < head[0]),  # opponent left
        (opp_head[0] > head[0]),  # opponent right
        (opp_head[1] < head[1]),  # opponent up
        (opp_head[1] > head[1])   # opponent down
    ]
    
    return np.array(state, dtype=np.float32)


def get_reward(game, my_agent, player_number, game_result, prev_length, prev_alive):
    """Calculate reward for the agent based on game state."""
    reward = 0.0
    
    # Reward for staying alive (survival bonus)
    if my_agent.alive:
        reward += 0.1
    else:
        reward -= 10.0  # Large penalty for dying
    
    # Reward for growing (length increase)
    length_increase = my_agent.length - prev_length
    reward += length_increase * 0.5
    
    # Reward for winning
    if game_result == GameResult.AGENT1_WIN and player_number == 1:
        reward += 100.0
    elif game_result == GameResult.AGENT2_WIN and player_number == 2:
        reward += 100.0
    
    # Penalty for losing
    if game_result == GameResult.AGENT2_WIN and player_number == 1:
        reward -= 50.0
    elif game_result == GameResult.AGENT1_WIN and player_number == 2:
        reward -= 50.0
    
    # Small penalty for draws
    if game_result == GameResult.DRAW:
        reward -= 10.0
    
    return reward


def get_action_from_model(model, state, epsilon=0.0):
    """Get action from model using epsilon-greedy policy."""
    if random.random() < epsilon:
        # Random exploration
        return random.randint(0, 3)
    else:
        # Exploitation - use model prediction
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            return q_values.argmax().item()


def direction_to_action(direction):
    """Convert Direction enum to action index."""
    if direction == Direction.LEFT:
        return 0
    elif direction == Direction.RIGHT:
        return 1
    elif direction == Direction.UP:
        return 2
    else:  # DOWN
        return 3


def action_to_direction(action):
    """Convert action index to Direction enum."""
    if action == 0:
        return Direction.LEFT
    elif action == 1:
        return Direction.RIGHT
    elif action == 2:
        return Direction.UP
    else:  # action == 3
        return Direction.DOWN


def random_action():
    """Get a random action."""
    return random.randint(0, 3)


def train_dqn(num_episodes=1000, batch_size=64, learning_rate=0.001, 
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
              memory_size=10000, target_update=10, save_interval=100):
    """
    Train the DQN model.
    
    Args:
        num_episodes: Number of training episodes
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        epsilon_start: Starting epsilon for epsilon-greedy
        epsilon_end: Final epsilon value
        epsilon_decay: Epsilon decay rate per episode
        memory_size: Size of replay memory
        target_update: Update target network every N episodes
        save_interval: Save model every N episodes
    """
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    state_size = 11
    action_size = 4
    
    # Main network and target network
    model = DQN(state_size, action_size).to(device)
    target_model = DQN(state_size, action_size).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Replay memory
    memory = deque(maxlen=memory_size)
    
    # Training statistics
    epsilon = epsilon_start
    scores = []
    wins = 0
    losses = 0
    draws = 0
    
    print("Starting training...")
    print(f"Episodes: {num_episodes}, Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    for episode in range(num_episodes):
        # Create new game
        game = Game()
        player_number = 1  # Train as player 1
        
        # Track previous state for reward calculation
        prev_length = game.agent1.length
        prev_alive = game.agent1.alive
        
        episode_reward = 0
        steps = 0
        max_steps = 200
        
        while steps < max_steps:
            # Get current state
            state = get_state(game, game.agent1, player_number)
            
            # Get action from model
            action = get_action_from_model(model, state, epsilon)
            direction = action_to_direction(action)
            
            # Get opponent action (random for training)
            opp_action = random_action()
            opp_direction = action_to_direction(opp_action)
            
            # Execute moves
            game_result = game.step(direction, opp_direction, boost1=False, boost2=False)
            
            # Calculate reward
            reward = get_reward(game, game.agent1, player_number, game_result, prev_length, prev_alive)
            episode_reward += reward
            
            # Get next state
            next_state = get_state(game, game.agent1, player_number) if game.agent1.alive else None
            
            # Store experience in replay memory
            if next_state is not None:
                memory.append((state, action, reward, next_state, False))
            else:
                memory.append((state, action, reward, state, True))  # Terminal state
            
            # Update previous state
            prev_length = game.agent1.length
            prev_alive = game.agent1.alive
            
            steps += 1
            
            # Check if game ended
            if game_result is not None:
                if game_result == GameResult.AGENT1_WIN:
                    wins += 1
                elif game_result == GameResult.AGENT2_WIN:
                    losses += 1
                else:
                    draws += 1
                break
        
        # Train the model
        if len(memory) > batch_size:
            # Sample random batch from memory
            batch = random.sample(memory, batch_size)
            
            states = torch.FloatTensor([e[0] for e in batch]).to(device)
            actions = torch.LongTensor([e[1] for e in batch]).to(device)
            rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
            next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
            dones = torch.BoolTensor([e[4] for e in batch]).to(device)
            
            # Current Q values
            current_q_values = model(states).gather(1, actions.unsqueeze(1))
            
            # Next Q values from target network
            with torch.no_grad():
                next_q_values = target_model(next_states).max(1)[0].detach()
                target_q_values = rewards + (0.99 * next_q_values * ~dones)
            
            # Compute loss
            loss = criterion(current_q_values.squeeze(), target_q_values)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Update target network
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Record statistics
        scores.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            win_rate = wins / (episode + 1) * 100
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_score:.2f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Wins: {wins} | Losses: {losses} | Draws: {draws} | "
                  f"Win Rate: {win_rate:.1f}%")
        
        # Save model
        if (episode + 1) % save_interval == 0:
            model_path = f"model_episode_{episode + 1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    
    # Save final model
    final_model_path = "model_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")
    print(f"Final statistics: Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"Final win rate: {wins / num_episodes * 100:.1f}%")
    
    return model


if __name__ == "__main__":
    # Training parameters
    NUM_EPISODES = 1000000000  # Training for 10,000 episodes for better performance
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    
    print("=" * 60)
    print("Case Closed Agent Training")
    print(f"Training for {NUM_EPISODES} episodes")
    print("=" * 60)
    
    model = train_dqn(
        num_episodes=NUM_EPISODES,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        save_interval= 500000  # Save every 500 episodes for long training runs
    )
    
    print("\nTraining finished! Use the saved model in agent.py")

