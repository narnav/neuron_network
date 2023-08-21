import numpy as np

# Define constants
EMPTY = 0
X = 1
O = -1
BOARD_SIZE = 3

# Initialize Q-values randomly
Q = np.random.rand(BOARD_SIZE ** 2, BOARD_SIZE ** 2)

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Function to convert 2D board coordinates to a flat index
def flatten_coordinates(row, col):
    return row * BOARD_SIZE + col

# Function to choose an action using epsilon-greedy policy
def choose_action(state):
    if not np.any(state == EMPTY):
        return None
    return np.random.choice(np.where(state == EMPTY)[0])


# Function to update Q-values using Q-learning
def update_q_value(state, action, reward, next_state):
    Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

# Function to check if a player has won
def is_winner(board, player):
    win_patterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]             # Diagonals
    ]
    return any(all(board[idx] == player for idx in pattern) for pattern in win_patterns)

# Simulate Tic-Tac-Toe game and train the neural network
def train(num_episodes):
    for episode in range(num_episodes):
        state = np.zeros(BOARD_SIZE ** 2, dtype=int)
        done = False
        
        while not done:
            action = choose_action(state)
            row, col = divmod(action, BOARD_SIZE)
            
            next_state = state.copy()
            if action < len(next_state):
                next_state[action] = X
            else:
                # Do nothing
                continue
            
            # Simulate opponent's move (randomly)
            available_actions = np.where(next_state == EMPTY)[0]
            if len(available_actions) > 0:
                # print(available_actions)
                opponent_action = np.random.choice(available_actions)
                next_state[opponent_action] = O
            
            # Check for game outcome and calculate reward
            reward = 0
            
            if is_winner(next_state, X):
                reward = 1
                done = True
            elif is_winner(next_state, O):
                reward = -1
                done = True
            elif len(available_actions) == 0:
                done = True

            if reward != 0:
                print(reward)
                print(state)
            update_q_value(flatten_coordinates(row, col), action, reward, flatten_coordinates(row, col))
            state = next_state



# Train the neural network
train(num_episodes=1000)
print(Q)
# Now you can use the Q-values to make decisions in the game
