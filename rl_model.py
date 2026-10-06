"""Q-Learning with SGDRegressor over the 10 x 10 environment."""

import random

import numpy as np
from sklearn.linear_model import SGDRegressor

from rl_environment import ACTION_NAMES, COLUMNS, GOAL, GRID, ROWS, START, step

EPISODES = 500
MAX_STEPS = 200
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99
LEARNING_RATE = 0.1
SEED = 42

NUMBER_OF_ACTIONS = len(ACTION_NAMES)
NUMBER_OF_FEATURES = ROWS * COLUMNS * NUMBER_OF_ACTIONS

# One-hot encoding: row i of this matrix is the feature vector of state-action pair i
FEATURES = np.eye(NUMBER_OF_FEATURES)

PARAMETERS = [
    {"name": "Training episodes", "value": EPISODES,
     "purpose": "Complete attempts from A. Enough for epsilon to reach its minimum and for the route to stabilise."},
    {"name": "Maximum steps per episode", "value": MAX_STEPS,
     "purpose": "Ends an episode that never reaches T. It is ten times the shortest route."},
    {"name": "Discount factor (γ)", "value": GAMMA,
     "purpose": "Weight of future rewards. Close to 1 so the +50 of T is still felt 20 steps away."},
    {"name": "Initial epsilon (ε)", "value": EPSILON_START,
     "purpose": "The agent starts knowing nothing, so it explores 100 % of the time."},
    {"name": "Minimum epsilon", "value": EPSILON_MIN,
     "purpose": "Keeps 5 % exploration until the end of training."},
    {"name": "Epsilon decay", "value": EPSILON_DECAY,
     "purpose": "Epsilon is multiplied by this value after each episode. It reaches the minimum around episode 300."},
    {"name": "Learning rate (eta0)", "value": LEARNING_RATE,
     "purpose": "How far each update moves a Q-value towards its target."},
    {"name": "Random seed", "value": SEED,
     "purpose": "Makes every training run reproducible."},
]


def state_index(state):
    return state[0] * COLUMNS + state[1]


def encode(state, action):
    return FEATURES[state_index(state) * NUMBER_OF_ACTIONS + action]


def predict_q_table(model):
    """Q-values of every cell: one row per state, one column per action."""
    return model.predict(FEATURES).reshape(ROWS * COLUMNS, NUMBER_OF_ACTIONS)


def choose_action(q_values, epsilon, rng):
    """Epsilon-greedy: explore with probability epsilon, otherwise take the best Q-value."""
    if rng.random() < epsilon:
        return rng.randrange(NUMBER_OF_ACTIONS)
    best_actions = np.flatnonzero(q_values == q_values.max()).tolist()
    return rng.choice(best_actions)


def train():
    rng = random.Random(SEED)
    model = SGDRegressor(
        loss="squared_error",
        penalty=None,
        fit_intercept=False,
        learning_rate="constant",
        eta0=LEARNING_RATE,
        random_state=SEED,
    )
    # A sample of zeros initialises the weights at 0 so predict() can be used
    model.partial_fit(np.zeros((1, NUMBER_OF_FEATURES)), np.array([0.0]))

    epsilon = EPSILON_START
    successes = 0
    episode_rewards = []

    for _ in range(EPISODES):
        q_table = predict_q_table(model)
        state = START
        total = 0
        features = []
        targets = []

        for _ in range(MAX_STEPS):
            action = choose_action(q_table[state_index(state)], epsilon, rng)
            next_state, reward, done, _ = step(state, action)

            # Q-Learning target: a terminal state has no future reward
            if done:
                target = reward
            else:
                target = reward + GAMMA * q_table[state_index(next_state)].max()

            features.append(encode(state, action))
            targets.append(target)
            total += reward
            state = next_state
            if done:
                successes += 1
                break

        # Q-value update with every transition observed in the episode
        model.partial_fit(np.array(features), np.array(targets))
        episode_rewards.append(total)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    q_table = predict_q_table(model)

    return {
        "episodes": EPISODES,
        "successes": successes,
        "success_rate": round(100 * successes / EPISODES, 1),
        "average_reward": round(float(np.mean(episode_rewards)), 2),
        "last_100_average": round(float(np.mean(episode_rewards[-100:])), 2),
        "final_epsilon": round(epsilon, 3),
        **evaluate(q_table),
        "q_values": q_values_table(q_table),
    }


def evaluate(q_table):
    """Follows the learned policy without exploration and records every step."""
    state = START
    path = [state]
    steps = []

    for number in range(1, MAX_STEPS + 1):
        action = int(np.argmax(q_table[state_index(state)]))
        next_state, reward, done, cell_type = step(state, action)
        steps.append({
            "number": number,
            "state": state,
            "action": ACTION_NAMES[action],
            "next_state": next_state,
            "cell_type": cell_type,
            "reward": reward,
        })
        path.append(next_state)
        state = next_state
        if done:
            break

    return {
        "reached_goal": state == GOAL,
        "moves": len(steps),
        "total_reward": sum(item["reward"] for item in steps),
        "dangers_entered": sum(item["cell_type"] == "Danger" for item in steps),
        "collisions": sum(item["cell_type"] in ("Wall", "Outside") for item in steps),
        "path": path,
        "steps": steps,
    }


def q_values_table(q_table):
    """Learned Q-values of every state the agent can occupy."""
    rows = []
    for row in range(ROWS):
        for column in range(COLUMNS):
            if GRID[row][column] in "#T":
                continue
            values = q_table[state_index((row, column))]
            best = int(np.argmax(values))
            rows.append({
                "state": (row, column),
                "symbol": GRID[row][column],
                "action_values": [round(float(value), 2) for value in values],
                "best_index": best,
                "best_action": ACTION_NAMES[best],
            })
    return rows


if __name__ == "__main__":
    result = train()
    for key in ("episodes", "successes", "success_rate", "average_reward", "final_epsilon",
                "reached_goal", "moves", "total_reward", "dangers_entered", "collisions"):
        print(f"{key}: {result[key]}")
    print("path:", " -> ".join(str(cell) for cell in result["path"]))
