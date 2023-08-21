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
    if np.random.rand() < epsilon:
        return np.random.choice(np.where(state == EMPTY)[0])
    else:
        return np.argmax(Q[state])

# Function to update Q-values using Q-learning
def update_q_value(state, action, reward, next_state):
    if action < 9:
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

# Function to check if a player has won
def is_winner(board, player):
    win_patterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]             # Diagonals
    ]
    return any(all(board[idx] == player for idx in pattern) for pattern in win_patterns)

# Simulate Tic-Tac-Toe game and use the trained Q-values
def play_game():
    board = np.zeros(BOARD_SIZE ** 2, dtype=int)
    current_player = X

    while True:
        if current_player == X:
            print_board(board.reshape((BOARD_SIZE, BOARD_SIZE)))
            action = choose_action(board)
            if action is None:
                print("It's a draw!")
                break
        else:
            print("Opponent's turn...")
            available_actions = np.where(board == EMPTY)[0]
            action = np.random.choice(available_actions)
        if action < BOARD_SIZE * BOARD_SIZE :
            board[action] = current_player
        if is_winner(board, current_player):
            print_board(board.reshape((BOARD_SIZE, BOARD_SIZE)))
            print(f"Player {current_player} wins!")
            break
        elif np.all(board != EMPTY):
            print_board(board.reshape((BOARD_SIZE, BOARD_SIZE)))
            print("It's a draw!")
            break

        update_q_value(action, action, 0, action)  # Update Q-values for the chosen action
        current_player = O if current_player == X else X

# Function to print the game board
def print_board(board):
    symbols = [" ", "X", "O"]
    for row in board:
        print(" | ".join([symbols[cell] for cell in row]))
        print("-" * 9)

if __name__ == "__main__":
    play_game()
