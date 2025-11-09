# Training Guide for Case Closed Agent

This guide explains how to train a neural network model for the Case Closed agent using Deep Q-Network (DQN) reinforcement learning.

## Overview

The training system uses:
- **DQN (Deep Q-Network)**: A neural network that learns to predict the best action (Q-value) for each game state
- **Experience Replay**: Stores past experiences and samples from them for training
- **Target Network**: A separate network used for stable Q-value estimation
- **Epsilon-Greedy Exploration**: Balances exploration (trying new actions) vs exploitation (using learned knowledge)

## Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   pip install torch numpy
   ```

2. **Run training**:
   ```bash
   python train.py
   ```

3. **Use trained model**: The trained model (`model_final.pth`) will be automatically loaded by `agent.py` when you run it.

## Training Process

### How It Works

1. **Game Simulation**: The training script plays games against a random opponent
2. **Experience Collection**: Each step (state, action, reward, next_state) is stored in replay memory
3. **Model Training**: Periodically, batches of experiences are sampled and used to train the neural network
4. **Target Network Updates**: The target network is updated periodically for stable learning
5. **Epsilon Decay**: Exploration rate decreases over time as the model learns

### Training Parameters

You can modify these in `train.py`:

- `num_episodes`: Number of training games (default: 500, increase for better performance)
- `batch_size`: Number of experiences per training batch (default: 64)
- `learning_rate`: How fast the model learns (default: 0.001)
- `epsilon_start`: Initial exploration rate (default: 1.0 = 100% random)
- `epsilon_end`: Final exploration rate (default: 0.01 = 1% random)
- `epsilon_decay`: How quickly exploration decreases (default: 0.995)
- `memory_size`: Size of experience replay buffer (default: 10000)
- `target_update`: How often to update target network (default: every 10 episodes)

### Reward System

The model learns from these rewards:
- **+0.1**: Staying alive each turn
- **+0.5 per cell**: Growing longer (length increase)
- **+100**: Winning the game
- **-50**: Losing the game
- **-10**: Draw
- **-10**: Dying

## Model Architecture

The neural network has:
- **Input**: 11 features (danger directions, current direction, opponent location)
- **Hidden Layers**: 3 layers with 128 neurons each
- **Output**: 4 Q-values (one for each action: LEFT, RIGHT, UP, DOWN)
- **Activation**: ReLU

## Training Tips

1. **More Episodes = Better Performance**: Start with 500 episodes, increase to 1000-5000 for better results
2. **Monitor Progress**: Watch the win rate and average reward increase over time
3. **Save Checkpoints**: Models are saved every 50 episodes (configurable)
4. **GPU Acceleration**: If you have a GPU, PyTorch will automatically use it
5. **Adjust Learning Rate**: If training is unstable, try lower learning rates (0.0001)

## Using the Trained Model

Once training completes:

1. The final model is saved as `model_final.pth`
2. `agent.py` automatically loads this model on startup
3. If no model is found, the agent falls back to heuristic-based play
4. The model makes decisions in real-time during games

## Advanced: Self-Play Training

For even better performance, you can modify `train.py` to:
- Train against a previous version of itself (self-play)
- Use a more sophisticated opponent (e.g., another trained model)
- Implement curriculum learning (start easy, get harder)

## Troubleshooting

**Model not loading?**
- Check that `model_final.pth` exists in the same directory as `agent.py`
- Verify PyTorch is installed: `pip install torch`

**Training is slow?**
- Reduce `num_episodes` for faster testing
- Use GPU if available (PyTorch will detect automatically)
- Reduce `memory_size` and `batch_size`

**Poor performance?**
- Increase training episodes
- Adjust reward weights in `get_reward()` function
- Try different network architectures (more/fewer layers)
- Experiment with learning rate and epsilon decay

## Example Training Output

```
Starting training...
Using device: cpu
Episodes: 500, Batch size: 64, Learning rate: 0.001
Episode 10/500 | Avg Reward: -5.23 | Epsilon: 0.951 | Wins: 2 | Losses: 7 | Draws: 1 | Win Rate: 20.0%
Episode 20/500 | Avg Reward: 2.15 | Epsilon: 0.904 | Wins: 5 | Losses: 12 | Draws: 3 | Win Rate: 25.0%
...
Episode 500/500 | Avg Reward: 45.32 | Epsilon: 0.082 | Wins: 312 | Losses: 145 | Draws: 43 | Win Rate: 62.4%
Training complete! Final model saved to model_final.pth
```

