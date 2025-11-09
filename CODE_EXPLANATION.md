# Code Flow and JSON Format Explanation

## Overview
This document explains how commands flow through the system and the exact JSON format sent to `agent.py`.

---

## Command Flow in `case_closed_game.py`

### 1. Direction Enum Definition
**Location:** `case_closed_game.py` lines 62-66

```python
class Direction(Enum):
    UP = (0, -1)      # Move up: x unchanged, y decreases
    DOWN = (0, 1)     # Move down: x unchanged, y increases
    RIGHT = (1, 0)    # Move right: x increases, y unchanged
    LEFT = (-1, 0)    # Move left: x decreases, y unchanged
```

Each direction is a tuple `(dx, dy)` representing the change in coordinates.

### 2. Command Execution Flow

#### Step 1: Agent Returns Move String
**Location:** `agent.py` line 109
- Agent returns JSON: `{"move": "DIRECTION"}` or `{"move": "DIRECTION:BOOST"}`
- Example: `{"move": "RIGHT"}` or `{"move": "RIGHT:BOOST"}`

#### Step 2: Judge Parses Move
**Location:** `judge_engine.py` lines 163-211

The `handle_move()` method:
1. **Parses the string** (line 172): Splits on `:` to separate direction and boost
   ```python
   move_parts = move.upper().split(':')
   direction_str = move_parts[0]  # e.g., "RIGHT"
   use_boost = len(move_parts) > 1 and move_parts[1] == 'BOOST'
   ```

2. **Converts to Direction enum** (lines 177-188):
   ```python
   direction_map = {
       'UP': Direction.UP,
       'DOWN': Direction.DOWN,
       'LEFT': Direction.LEFT,
       'RIGHT': Direction.RIGHT,
   }
   direction = direction_map[direction_str]
   ```

3. **Validates move** (lines 194-201): Checks if trying to move opposite direction
   - If invalid, uses current direction instead

4. **Returns tuple** (line 211): `(True, use_boost, direction)`

#### Step 3: Game.step() Called
**Location:** `judge_engine.py` line 330
```python
result = judge.game.step(p1_direction, p2_direction, p1_boost, p2_boost)
```

**Location:** `case_closed_game.py` lines 181-208

The `step()` method:
- Takes two `Direction` enums and two boolean boost flags
- Calls `agent.move()` for each agent (lines 195-196)

#### Step 4: Agent.move() Executes
**Location:** `case_closed_game.py` lines 91-161

The `move()` method processes the command:

1. **Determines number of moves** (line 111):
   ```python
   num_moves = 2 if use_boost else 1
   ```

2. **For each move** (line 117):
   - Gets current head position (line 124): `head = self.trail[-1]`
   - Calculates new position (lines 125-126):
     ```python
     dx, dy = direction.value  # e.g., Direction.RIGHT.value = (1, 0)
     new_head = (head[0] + dx, head[1] + dy)
     ```
   
3. **Applies torus (wraparound)** (line 128):
   ```python
   new_head = self.board._torus_check(new_head)
   ```
   - Location: `case_closed_game.py` lines 21-25
   - Wraps coordinates: `x % width`, `y % height`

4. **Checks collision** (lines 130-153):
   - If cell is `AGENT` (occupied):
     - If hits own trail → agent dies (line 139)
     - If hits other agent's head → both die (head-on collision, line 147)
     - If hits other agent's trail → agent dies (line 152)

5. **Executes move** (lines 157-159):
   ```python
   self.trail.append(new_head)  # Add new position to trail
   self.length += 1              # Increase trail length
   self.board.set_cell_state(new_head, AGENT)  # Mark cell as occupied
   ```

### 3. Coordinate System
- **Origin (0,0)**: Top-left corner
- **X-axis**: Increases rightward
- **Y-axis**: Increases downward
- **Torus wrapping**: Coordinates wrap around edges (18x20 board)

---

## Exact JSON Format Sent to `agent.py`

### Endpoint: `/send-state` (POST)
**Location:** `judge_engine.py` lines 76-98

The judge sends this JSON structure:

```json
{
  "board": [
    [0, 0, 0, 0, ...],  // 18 rows, each with 20 elements
    [0, 0, 0, 0, ...],  // 0 = EMPTY, 1 = AGENT
    ...
  ],
  "agent1_trail": [
    [1, 2],    // First position (x, y)
    [2, 2],    // Second position
    [3, 2],    // ... and so on
    ...
  ],
  "agent2_trail": [
    [17, 15],
    [16, 15],
    ...
  ],
  "agent1_length": 5,           // Integer: current trail length
  "agent2_length": 5,            // Integer: current trail length
  "agent1_alive": true,          // Boolean: whether agent is alive
  "agent2_alive": true,          // Boolean: whether agent is alive
  "agent1_boosts": 3,            // Integer: remaining boosts (0-3)
  "agent2_boosts": 3,            // Integer: remaining boosts (0-3)
  "turn_count": 42,              // Integer: current turn number
  "player_number": 1             // Integer: 1 or 2 (which player this is for)
}
```

### Field Details:

1. **`board`**: 2D array (18 rows × 20 columns)
   - `0` = `EMPTY` (unoccupied cell)
   - `1` = `AGENT` (occupied by any agent's trail)
   - Type: `List[List[int]]`

2. **`agent1_trail`**: List of `[x, y]` coordinate pairs
   - Ordered from oldest to newest position
   - Last element is the current head position
   - Type: `List[List[int]]`

3. **`agent2_trail`**: Same format as `agent1_trail`

4. **`agent1_length`**: Integer representing trail length
   - Starts at 2, increases by 1 each move
   - Type: `int`

5. **`agent2_length`**: Same as `agent1_length`

6. **`agent1_alive`**: Boolean indicating if agent is alive
   - `true` = alive, `false` = crashed
   - Type: `bool`

7. **`agent2_alive`**: Same as `agent1_alive`

8. **`agent1_boosts`**: Integer (0-3) for remaining speed boosts
   - Starts at 3, decreases by 1 when boost is used
   - Type: `int`

9. **`agent2_boosts`**: Same as `agent1_boosts`

10. **`turn_count`**: Integer representing current turn number
    - Starts at 0, increments each turn
    - Type: `int`

11. **`player_number`**: Integer (1 or 2) indicating which player this state is for
    - Type: `int`

### Example JSON Payload:

```json
{
  "board": [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ... (14 more rows)
  ],
  "agent1_trail": [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2]],
  "agent2_trail": [[17, 15], [16, 15], [15, 15], [14, 15], [13, 15]],
  "agent1_length": 5,
  "agent2_length": 5,
  "agent1_alive": true,
  "agent2_alive": true,
  "agent1_boosts": 2,
  "agent2_boosts": 3,
  "turn_count": 3,
  "player_number": 1
}
```

### How `agent.py` Receives This:

**Location:** `agent.py` lines 67-77

```python
@app.route("/send-state", methods=["POST"])
def receive_state():
    data = request.get_json()  # Receives the JSON above
    _update_local_game_from_post(data)  # Updates GLOBAL_GAME
    return jsonify({"status": "state received"}), 200
```

The `_update_local_game_from_post()` function (lines 30-64) extracts each field and updates the local game state.

---

## Move Request Format

### Endpoint: `/send-move` (GET)
**Location:** `judge_engine.py` lines 100-129

The judge sends query parameters:
- `player_number`: 1 or 2
- `attempt_number`: 1 or 2 (retry attempt)
- `random_moves_left`: Number of random moves remaining (starts at 5)
- `turn_count`: Current turn number

**Example request:**
```
GET /send-move?player_number=1&attempt_number=1&random_moves_left=5&turn_count=42
```

**Expected response from agent:**
```json
{
  "move": "RIGHT"
}
```

or with boost:
```json
{
  "move": "RIGHT:BOOST"
}
```

---

## Complete Command Flow Diagram

```
1. Judge calls GET /send-move
   ↓
2. agent.py returns {"move": "RIGHT:BOOST"}
   ↓
3. judge_engine.py handle_move() parses string
   - Splits "RIGHT:BOOST" → direction="RIGHT", boost=True
   - Converts to Direction.RIGHT enum
   - Validates (checks opposite direction)
   ↓
4. judge_engine.py calls game.step(Direction.RIGHT, Direction.LEFT, True, False)
   ↓
5. case_closed_game.py Game.step() calls:
   - agent1.move(Direction.RIGHT, other_agent, use_boost=True)
   - agent2.move(Direction.LEFT, other_agent, use_boost=False)
   ↓
6. case_closed_game.py Agent.move() executes:
   - Calculates new_head = (current_x + dx, current_y + dy)
   - Applies torus wrapping
   - Checks collisions
   - Updates trail and board state
   ↓
7. Judge sends updated state via POST /send-state
   ↓
8. Repeat from step 1
```

---

## Key Code Locations Summary

| Action | File | Line(s) |
|--------|------|---------|
| Direction enum definition | `case_closed_game.py` | 62-66 |
| Direction to coordinates | `case_closed_game.py` | 125-126 |
| Move execution | `case_closed_game.py` | 91-161 |
| Game step | `case_closed_game.py` | 181-208 |
| JSON state creation | `judge_engine.py` | 80-92 |
| Move parsing | `judge_engine.py` | 163-211 |
| JSON state reception | `agent.py` | 67-77 |
| Move response | `agent.py` | 80-109 |

