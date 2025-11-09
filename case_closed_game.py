import random
from collections import deque
from enum import Enum
from typing import Optional

# Constants for cell states on the game board
EMPTY = 0  # Empty cell that agents can move into
AGENT = 1  # Cell occupied by an agent's trail

"""
GameBoard class manages the game board.

Handles the 2D grid, state of each cell, and provides torus (wraparound)
functionality for all coordinate-based operations.
"""
class GameBoard:
    def __init__(self, height: int = 18, width: int = 20):
        """
        Initialize the game board with specified dimensions.
        
        Args:
            height: Height of the game board (default: 18)
            width: Width of the game board (default: 20)
        """
        self.height = height  # Height of the game board (18)
        self.width = width    # Width of the game board (20)
        # Create a 2D grid initialized with all empty cells
        self.grid = [[EMPTY for _ in range(width)] for _ in range(height)]

    def _torus_check(self, position: tuple[int, int]) -> tuple[int, int]:
        """
        Normalizes coordinates to handle torus (wraparound) topology.
        If coordinates go out of bounds, they wrap around to the other side.
        
        Args:
            position: (x, y) tuple of coordinates
            
        Returns:
            Normalized (x, y) tuple within board bounds
        """
        x, y = position
        normalized_x = x % self.width   # Wrap x coordinate
        normalized_y = y % self.height  # Wrap y coordinate
        return (normalized_x, normalized_y)
    
    def get_cell_state(self, position: tuple[int, int]) -> int:
        """
        Gets the state of a cell at the given position.
        
        Args:
            position: (x, y) tuple of coordinates
            
        Returns:
            Cell state (EMPTY or AGENT)
        """
        x, y = self._torus_check(position)  # Normalize coordinates first
        return self.grid[y][x]

    def set_cell_state(self, position: tuple[int, int], state: int):
        """
        Sets the state of a cell at the given position.
        
        Args:
            position: (x, y) tuple of coordinates
            state: New state to set (EMPTY or AGENT)
        """
        x, y = self._torus_check(position)  # Normalize coordinates first
        self.grid[y][x] = state

    def get_random_empty_cell(self) -> tuple[int, int] | None:
        """
        Finds a random empty cell on the board.
        
        Returns:
            (x, y) tuple of an empty cell, or None if no empty cells exist
        """
        empty_cells = []
        # Collect all empty cell positions
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == EMPTY:
                    empty_cells.append((x, y))
        
        if not empty_cells:
            return None
        
        return random.choice(empty_cells)

    def __str__(self) -> str:
        """
        String representation of the board for visualization.
        '.' represents empty cells, 'A' represents agent cells.
        """
        chars = {EMPTY: '.', AGENT: 'A'}
        board_str = ""
        for y in range(self.height):
            for x in range(self.width):
                board_str += chars.get(self.grid[y][x], '?') + ' '
            board_str += '\n'
        return board_str


# Direction constants (legacy, kept for compatibility)
UP = (0, -1)
DOWN = (0, 1)
RIGHT = (1, 0)
LEFT = (-1, 0)

class Direction(Enum):
    """
    Enumeration of possible movement directions.
    Values are (dx, dy) tuples representing movement deltas.
    """
    UP = (0, -1)      # Move up (decrease y)
    DOWN = (0, 1)     # Move down (increase y)
    RIGHT = (1, 0)    # Move right (increase x)
    LEFT = (-1, 0)    # Move left (decrease x)

class GameResult(Enum):
    """
    Enumeration of possible game outcomes.
    """
    AGENT1_WIN = 1  # Agent 1 wins the game
    AGENT2_WIN = 2  # Agent 2 wins the game
    DRAW = 3        # Game ends in a draw

class Agent:
    """
    This class represents an agent in the game. 
    It manages the agent's trail using a deque and handles movement logic.
    """
    def __init__(self, agent_id: str, start_pos: tuple[int, int], start_dir: Direction, board: GameBoard):
        """
        Initialize an agent with starting position and direction.
        
        Args:
            agent_id: Unique identifier for the agent
            start_pos: (x, y) starting position
            start_dir: Initial direction the agent is facing
            board: Reference to the game board
        """
        self.agent_id = agent_id
        # Calculate second position based on starting direction (agent starts with 2 cells)
        second = (start_pos[0] + start_dir.value[0], start_pos[1] + start_dir.value[1])
        self.trail = deque([start_pos, second])  # Trail of positions (deque for efficient append/pop)
        self.direction = start_dir  # Current facing direction
        self.board = board  # Reference to the game board
        self.alive = True  # Whether the agent is still alive
        self.length = 2  # Initial length of the trail (starts with 2 cells)
        self.boosts_remaining = 3  # Each agent gets 3 speed boosts

        # Mark initial positions on the board
        self.board.set_cell_state(start_pos, AGENT)
        self.board.set_cell_state(second, AGENT)
    
    def is_head(self, position: tuple[int, int]) -> bool:
        """
        Check if the given position is the agent's head (current position).
        
        Args:
            position: (x, y) position to check
            
        Returns:
            True if position is the agent's head, False otherwise
        """
        return position == self.trail[-1]  # Head is the last element in the trail
    
    def move(self, direction: Direction, other_agent: Optional['Agent'] = None, use_boost: bool = False) -> bool:
        """
        Moves the agent in the given direction and handles collisions.
        Agents leave a permanent trail behind them (trail never shrinks).
        
        Args:
            direction: Direction enum indicating where to move (UP, DOWN, LEFT, RIGHT)
            other_agent: The other agent on the board (for collision detection)
            use_boost: If True and boosts available, moves twice instead of once
        
        Returns:
            True if the agent survives the move, False if it dies
        """
        # If agent is already dead, can't move
        if not self.alive:
            return False

        # Boost logic (currently commented out but structure exists)
        # if use_boost and self.boosts_remaining <= 0:
        #     print(f'Agent {self.agent_id} tried to boost but has no boosts remaining')
        #     use_boost = False
        
        # Determine number of moves: 2 if boosting, 1 otherwise
        num_moves = 2 if use_boost else 1
        
        # Boost usage tracking (currently commented out)
        # if use_boost:
        #     self.boosts_remaining -= 1
        #     print(f'Agent {self.agent_id} used boost! ({self.boosts_remaining} remaining)')
        
        # Execute the move(s) - loop handles boost (2 moves) or normal (1 move)
        for move_num in range(num_moves):
            # Get current and requested direction vectors
            cur_dx, cur_dy = self.direction.value
            req_dx, req_dy = direction.value
            
            # Prevent moving directly backwards (invalid move)
            if (req_dx, req_dy) == (-cur_dx, -cur_dy):
                print('invalid move')
                continue  # Skip this move if invalid direction
            
            # Calculate new head position
            head = self.trail[-1]  # Current head position
            dx, dy = direction.value
            new_head = (head[0] + dx, head[1] + dy)
            
            # Normalize coordinates for torus (wraparound) topology
            new_head = self.board._torus_check(new_head)
            
            # Check what's in the target cell
            cell_state = self.board.get_cell_state(new_head)
            
            # Update agent's facing direction
            self.direction = direction
            
            # ===== COLLISION DETECTION =====
            # Handle collision with agent trail (own or opponent's)
            if cell_state == AGENT:
                # Check if it's our own trail (any part of our trail)
                if new_head in self.trail:
                    # Hit our own trail - agent dies
                    self.alive = False
                    return False
                
                # Check collision with the other agent
                if other_agent and other_agent.alive and new_head in other_agent.trail:
                    # Check for head-on collision (both agents moving into same cell)
                    if other_agent.is_head(new_head):
                        # Head-on collision: always a draw (both agents die)
                        self.alive = False
                        other_agent.alive = False
                        return False
                    else:
                        # Hit other agent's trail (not head-on) - this agent dies
                        self.alive = False
                        return False
            
            # ===== NORMAL MOVE (NO COLLISION) =====
            # Empty cell - safe to move into
            # Add new head to trail, trail keeps growing (never shrinks)
            self.trail.append(new_head)
            self.length += 1
            # Mark the new cell as occupied by agent
            self.board.set_cell_state(new_head, AGENT)
        
        return True  # Agent survived the move

    def get_trail_positions(self) -> list[tuple[int, int]]:
        """
        Returns a list of all positions in the agent's trail.
        
        Returns:
            List of (x, y) tuples representing all trail positions
        """
        return list(self.trail)
    

class Game:
    """
    Main game class that manages the game state, agents, and turn progression.
    """
    def __init__(self):
        """
        Initialize a new game with two agents on the board.
        """
        self.board = GameBoard()  # Create the game board
        # Initialize Agent 1 at position (1, 2) facing RIGHT
        self.agent1 = Agent(agent_id=1, start_pos=(1, 2), start_dir=Direction.RIGHT, board=self.board)
        # Initialize Agent 2 at position (17, 15) facing LEFT
        self.agent2 = Agent(agent_id=2, start_pos=(17, 15), start_dir=Direction.LEFT, board=self.board)
        self.turns = 0  # Track number of turns elapsed
    
    def reset(self):
        """
        Resets the game to the initial state.
        Creates a new board and reinitializes both agents.
        """
        self.board = GameBoard()  # Create fresh board
        # Reinitialize Agent 1 at position (1, 2) facing RIGHT
        self.agent1 = Agent(agent_id=1, start_pos=(1, 2), start_dir=Direction.RIGHT, board=self.board)
        # Reinitialize Agent 2 at position (17, 15) facing LEFT
        self.agent2 = Agent(agent_id=2, start_pos=(17, 15), start_dir=Direction.LEFT, board=self.board)
        self.turns = 0  # Reset turn counter
    
    def step(self, dir1: Direction, dir2: Direction, boost1: bool = False, boost2: bool = False):
        """
        ===== THIS IS WHERE AGENT MOVES ARE INPUT =====
        Advances the game by one step, moving both agents.
        
        This is the main method where agents provide their move decisions:
        - dir1: Direction for Agent 1 to move (UP, DOWN, LEFT, or RIGHT)
        - dir2: Direction for Agent 2 to move (UP, DOWN, LEFT, or RIGHT)
        - boost1: Whether Agent 1 wants to use a speed boost (moves twice)
        - boost2: Whether Agent 2 wants to use a speed boost (moves twice)
        
        Args:
            dir1: Movement direction for Agent 1
            dir2: Movement direction for Agent 2
            boost1: Whether Agent 1 uses boost (optional, default: False)
            boost2: Whether Agent 2 uses boost (optional, default: False)
        
        Returns:
            GameResult enum indicating the game outcome, or None if game continues
        """
        # Check if maximum turns reached (200 turn limit)
        if self.turns >= 200:
            print("Max turns reached. Checking trail lengths...")
            # Winner is determined by longest trail
            if self.agent1.length > self.agent2.length:
                print(f"Agent 1 wins with trail length {self.agent1.length} vs {self.agent2.length}")
                return GameResult.AGENT1_WIN
            elif self.agent2.length > self.agent1.length:
                print(f"Agent 2 wins with trail length {self.agent2.length} vs {self.agent1.length}")
                return GameResult.AGENT2_WIN
            else:
                print(f"Draw - both agents have trail length {self.agent1.length}")
                return GameResult.DRAW
        
        # ===== AGENT MOVES ARE EXECUTED HERE =====
        # Execute Agent 1's move with the provided direction and boost option
        agent_one_alive = self.agent1.move(dir1, other_agent=self.agent2, use_boost=boost1)
        # Execute Agent 2's move with the provided direction and boost option
        agent_two_alive = self.agent2.move(dir2, other_agent=self.agent1, use_boost=boost2)

        # ===== CHECK GAME END CONDITIONS =====
        # Determine game outcome based on agent survival
        if not agent_one_alive and not agent_two_alive:
            # Both agents crashed (draw)
            print("Both agents have crashed.")
            return GameResult.DRAW
        elif not agent_one_alive:
            # Agent 1 crashed, Agent 2 wins
            print("Agent 1 has crashed.")
            return GameResult.AGENT2_WIN
        elif not agent_two_alive:
            # Agent 2 crashed, Agent 1 wins
            print("Agent 2 has crashed.")
            return GameResult.AGENT1_WIN

        # Game continues - increment turn counter
        self.turns += 1
        return None  # Game is still ongoing
