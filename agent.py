import os
import uuid
import torch
import torch.nn as nn
import random
import numpy as np
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"

# Neural Network Model (same architecture as training)
class DQN(nn.Module):
    """Deep Q-Network for learning optimal actions in Case Closed game."""
    
    def __init__(self, state_size=11, action_size=4, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load trained model if available
MODEL_PATH = "model_final.pth"
MODEL = None
if os.path.exists(MODEL_PATH):
    try:
        MODEL = DQN(state_size=11, action_size=4)
        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        MODEL.eval()
        print(f"Loaded trained model from {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        MODEL = None
else:
    print(f"No trained model found at {MODEL_PATH}. Using heuristic-based approach.")


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


def get_state(game, my_agent, player_number):
    """Get state representation adapted from snake_agent for Case Closed game."""
    head = my_agent.trail[-1]  # Current head position
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
            # Check if it's our own trail (not the head)
            if pos in my_agent.trail and pos != head:
                return True
            # Check if it's opponent's trail
            if other_agent.alive and pos in other_agent.trail:
                return True
        return False
    
    # Get opponent's head position for state features
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
        
        # Opponent location (simplified - check if opponent is to left/right/up/down)
        (opp_head[0] < head[0]),  # opponent left
        (opp_head[0] > head[0]),  # opponent right
        (opp_head[1] < head[1]),  # opponent up
        (opp_head[1] > head[1])   # opponent down
    ]
    
    return np.array(state, dtype=int)


def get_action(state, n_games=0, use_model=True):
    """Get action using model (if available) or epsilon-greedy heuristic policy.
    
    Args:
        state: State representation array
        n_games: Number of games played (for epsilon decay)
        use_model: Whether to use trained model if available
    
    Returns:
        Action array [LEFT, RIGHT, UP, DOWN] with one-hot encoding
    """
    final_move = [0, 0, 0, 0]  # [LEFT, RIGHT, UP, DOWN]
    
    # Use trained model if available
    if use_model and MODEL is not None:
        try:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = MODEL(state_tensor)
                action = q_values.argmax().item()
                final_move[action] = 1
                return final_move
        except Exception as e:
            print(f"Error using model: {e}, falling back to heuristic")
    
    # Fallback to heuristic-based approach
    # Epsilon-greedy exploration
    epsilon = max(0, 80 - n_games)
    
    if random.randint(0, 200) < epsilon:
        # Random exploration
        move = random.randint(0, 3)
        final_move[move] = 1
    else:
        # Exploitation - use simple heuristic
        # Prefer safe moves (avoid collisions)
        if not state[0]:  # Not dangerous straight ahead
            # Continue in current direction
            if state[3]:  # dir_l
                final_move[0] = 1  # LEFT
            elif state[4]:  # dir_r
                final_move[1] = 1  # RIGHT
            elif state[5]:  # dir_u
                final_move[2] = 1  # UP
            else:  # dir_d
                final_move[3] = 1  # DOWN
        elif not state[1]:  # Not dangerous right
            # Turn right
            if state[3]:  # dir_l -> turn to UP
                final_move[2] = 1
            elif state[4]:  # dir_r -> turn to DOWN
                final_move[3] = 1
            elif state[5]:  # dir_u -> turn to RIGHT
                final_move[1] = 1
            else:  # dir_d -> turn to LEFT
                final_move[0] = 1
        elif not state[2]:  # Not dangerous left
            # Turn left
            if state[3]:  # dir_l -> turn to DOWN
                final_move[3] = 1
            elif state[4]:  # dir_r -> turn to UP
                final_move[2] = 1
            elif state[5]:  # dir_u -> turn to LEFT
                final_move[0] = 1
            else:  # dir_d -> turn to RIGHT
                final_move[1] = 1
        else:
            # All directions dangerous, pick random
            move = random.randint(0, 3)
            final_move[move] = 1
    
    return final_move


def should_use_boost(game, my_agent, player_number, state_array, turn_count, boosts_remaining):
    """Decide whether to use a boost based on game state and strategy.
    
    According to case_closed_game.py rules:
    - Boosts allow the agent to move twice in one turn
    - Each agent starts with 3 boosts
    - Useful for escaping danger, cutting off opponents, or gaining territory
    
    Args:
        game: The Game object
        my_agent: The current agent
        player_number: 1 or 2
        state_array: The state representation array
        turn_count: Current turn number
        boosts_remaining: Number of boosts left
        
    Returns:
        True if boost should be used, False otherwise
    """
    # Can't boost if no boosts remaining
    if boosts_remaining <= 0:
        return False
    
    # Get opponent agent
    other_agent = game.agent2 if player_number == 1 else game.agent1
    
    # Strategy 1: Boost to escape immediate danger
    # If all directions are dangerous, boost might help escape
    danger_straight = state_array[0]
    danger_right = state_array[1]
    danger_left = state_array[2]
    
    # If in high danger (2+ directions blocked), boost to escape
    danger_count = sum([danger_straight, danger_right, danger_left])
    if danger_count >= 2:
        return True
    
    # Strategy 2: Boost when opponent is close and we're in a good position
    # Calculate distance to opponent
    my_head = my_agent.trail[-1]
    opp_head = other_agent.trail[-1] if other_agent.alive else my_head
    
    # Manhattan distance with torus wrapping
    dx = min(abs(opp_head[0] - my_head[0]), 
             game.board.width - abs(opp_head[0] - my_head[0]))
    dy = min(abs(opp_head[1] - my_head[1]), 
             game.board.height - abs(opp_head[1] - my_head[1]))
    distance = dx + dy
    
    # If opponent is close (within 5 cells) and we have safe space ahead, boost
    if distance <= 5 and not danger_straight:
        # Boost to gain territory or cut off opponent
        return True
    
    # Strategy 3: Boost in early game to claim territory (first 30 turns)
    # Use 1 boost early to establish position
    if turn_count < 30 and boosts_remaining == 3 and not danger_straight:
        return True
    
    # Strategy 4: Boost when behind in trail length to catch up
    my_length = my_agent.length
    opp_length = other_agent.length if other_agent.alive else 0
    if opp_length > my_length + 3 and not danger_straight:
        # Boost to catch up when significantly behind
        return True
    
    # Strategy 5: Boost in late game if we have boosts left (last 20 turns)
    # Don't waste boosts - use them before game ends
    if turn_count >= 180 and boosts_remaining > 0 and not danger_straight:
        return True
    
    return False


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining
   
    # -----------------your code here-------------------
    # Get current game state
    state_old = get_state(GLOBAL_GAME, my_agent, player_number)
    
    # Get action using epsilon-greedy policy (adapted from snake_agent)
    n_games = 0  # Could track this if needed
    final_move = get_action(state_old, n_games)
    
    # Convert action array to direction string
    if final_move[0] == 1:  # LEFT
        move = "LEFT"
    elif final_move[1] == 1:  # RIGHT
        move = "RIGHT"
    elif final_move[2] == 1:  # UP
        move = "UP"
    else:  # DOWN
        move = "DOWN"
    
    # Decide whether to use boost based on strategic rules
    turn_count = state.get("turn_count", 0)
    use_boost = should_use_boost(
        GLOBAL_GAME, 
        my_agent, 
        player_number, 
        state_old, 
        turn_count, 
        boosts_remaining
    )
    
    if use_boost:
        move = move + ":BOOST"
    # -----------------end code here--------------------

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
