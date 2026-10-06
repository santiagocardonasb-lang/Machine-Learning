"""10 x 10 grid environment for the Reinforcement Learning module."""

# A = agent start, T = target, o = available path, # = wall, D = danger zone
GRID = [
    "Aoooo#oooo",
    "o##oD#o#oo",
    "oo#ooooDoo",
    "#oDo##oo#o",
    "oooDo#Dooo",
    "o##oooo#Do",
    "oooo#ooo#o",
    "D#oo#oDooo",
    "oooDoo#o#D",
    "oooooooooT",
]

ROWS = len(GRID)
COLUMNS = len(GRID[0])

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["Up", "Down", "Left", "Right"]


def find(symbol):
    for row in range(ROWS):
        for column in range(COLUMNS):
            if GRID[row][column] == symbol:
                return (row, column)


START = find("A")
GOAL = find("T")

REWARDS = {
    "Path": -1,
    "Outside": -3,
    "Wall": -5,
    "Danger": -15,
    "Target": 50,
}

CELL_TYPES = {"A": "Start", "o": "Path", "D": "Danger", "T": "Target"}


def step(state, action):
    """Applies one action. Returns the next state, the reward, whether the episode ended and the cell type."""
    row = state[0] + ACTIONS[action][0]
    column = state[1] + ACTIONS[action][1]

    # Invalid moves keep the agent where it was
    if not (0 <= row < ROWS and 0 <= column < COLUMNS):
        return state, REWARDS["Outside"], False, "Outside"
    if GRID[row][column] == "#":
        return state, REWARDS["Wall"], False, "Wall"

    cell_type = CELL_TYPES[GRID[row][column]]
    if cell_type == "Target":
        return (row, column), REWARDS["Target"], True, cell_type
    if cell_type == "Danger":
        return (row, column), REWARDS["Danger"], False, cell_type
    return (row, column), REWARDS["Path"], False, cell_type


def count(symbol):
    return sum(row.count(symbol) for row in GRID)


LEGEND = [
    {"symbol": "A", "name": "Agent start", "css": "cell-start", "count": count("A"),
     "meaning": "Where every episode begins"},
    {"symbol": "T", "name": "Target", "css": "cell-target", "count": count("T"),
     "meaning": "Reaching it ends the episode with the biggest reward"},
    {"symbol": "o", "name": "Available path", "css": "cell-path", "count": count("o"),
     "meaning": "Free cell, the agent can move through it"},
    {"symbol": "#", "name": "Wall", "css": "cell-wall", "count": count("#"),
     "meaning": "Obstacle, it cannot be crossed"},
    {"symbol": "D", "name": "Danger zone", "css": "cell-danger", "count": count("D"),
     "meaning": "The agent can enter it, but receives a heavy penalty"},
]

CELL_CSS = {item["symbol"]: item["css"] for item in LEGEND}

REWARD_TABLE = [
    {"event": "Move to a valid normal cell (o)", "reward": REWARDS["Path"],
     "position": "Moves", "ends": "No"},
    {"event": "Invalid move outside the grid", "reward": REWARDS["Outside"],
     "position": "Stays", "ends": "No"},
    {"event": "Hit a wall (#)", "reward": REWARDS["Wall"],
     "position": "Stays", "ends": "No"},
    {"event": "Enter a danger zone (D)", "reward": REWARDS["Danger"],
     "position": "Moves", "ends": "No"},
    {"event": "Reach the target (T)", "reward": REWARDS["Target"],
     "position": "Moves", "ends": "Yes"},
]
