import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult, AGENT

import numpy
import random
import torch
from model import Linear_QNet, QTrainer
from helper import plot



# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"


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
    
    MAX_MEMORY = 100_000
    BATCH_SIZE = 1000
    LR = 0.001

    class Agent:

        def __init__(self):
            self.n_games = 0
            self.epsilon = 0 # randomness
            self.gamma = 0.9 # discount rate (smaller than 1)
            self.memory = deque(maxlen=MAX_MEMORY) # popleft()
            self.model = Linear_QNet(7, 256, 3)
            self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        def get_state(self, game) -> numpy.ndarray:
            head = Agent.agent1.trail[-1]
            point_left = (head[0] - 1, head[1])
            point_right = (head[0] + 1, head[1])
            point_up = (head[0], head[1] - 1)
            point_down = (head[0], head[1] + 1)

            dir_left = Agent.agent1.direction == Direction.LEFT
            dir_right = Agent.agent1.direction == Direction.RIGHT
            dir_up = Agent.agent1.direction == Direction.UP
            dir_down = Agent.agent1.direction == Direction.DOWN

            # Helper function to check collision using available Game methods
            def is_collision(point):
                return game.board.get_cell_state(point) == AGENT

            state = [
                # Danger straight
                (dir_right and is_collision(point_right)) or
                (dir_left and is_collision(point_left)) or
                (dir_up and is_collision(point_up)) or
                (dir_down and is_collision(point_down)) or

                # Danger right
                (dir_down and is_collision(point_right)) or
                (dir_up and is_collision(point_left)) or
                (dir_right and is_collision(point_down)) or
                (dir_left and is_collision(point_up)) or

                # Danger left
                (dir_up and is_collision(point_right)) or
                (dir_down and is_collision(point_left)) or
                (dir_left and is_collision(point_up)) or
                (dir_right and is_collision(point_down)) or

                # Move direction
                dir_left,
                dir_right,
                dir_up,
                dir_down
            ]
            
            return numpy.array(state, dtype=int)

        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done)) # popleft if max_memory is reached


        def train_long_memory(self):
            if len(self.memory) > BATCH_SIZE:
                mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
            else:
                mini_sample = self.memory

            states, actions, rewards, next_states, done = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states, done)

        def train_short_memory(self, state, action, reward, next_state, done):
            self.trainer.train_step(state, action, reward, next_state, done)
        
        def get_action(self, state) -> list[int]:
            # random moves: tradeoff exploration / exploitation

            self.epsilon = 80 - self.n_games
            final_move = [0,0]
            if random.randint(0, 200) < self.epsilon:
                move = random.randint(0, 1)
                final_move[move] = 1
            else:
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
                final_move[move] = 1
            
            return final_move

    move = Agent.get_action(state)


    def train():
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        agent = Agent()
        game = Game()

        while True:
            # get old state
            old_state = agent.get_state(game)

            # get move
            final_move = agent.get_action(old_state)

            # perform move and get new state
            reward, done, score = game.step(final_move)
            new_state = agent.get_state(game)

            # train short memort
            agent.train_short_memory(old_state, final_move, reward, new_state, done)

            # train long memory
            agent.train_long_memory(old_state, final_move, reward, new_state, done)

            if done:
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                print('Game', agent.n_games, 'Score', 'Record', record)
                
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

    return jsonify({"move": move}), 200




    # -----------------end code here--------------------

    




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
