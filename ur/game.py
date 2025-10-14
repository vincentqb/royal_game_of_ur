import numpy as np

N_PLAYER = 2
N_PIECE = 7
N_BOARD = 14
SAFE = sorted([8])
ROSETTE = sorted([4, 8, 14])
COMMON = sorted(range(5, 12 + 1))


def create_board():
    # pieces start on [0] and end on [-1]
    board = np.zeros((N_PLAYER, N_BOARD + 2), dtype=np.uint8)
    board[:, 0] = N_PIECE
    return board


def standardize_state(board, player):
    # return board
    # return np.concat([np.array([player], dtype=np.uint8), board.flatten()], dtype=np.uint8)
    rows = list(range(N_PLAYER))
    rows = rows[player:] + rows[:player]  # rotate
    return board[rows, :]


def execute_move(board, player, start, end):
    enemies = [p for p in range(N_PLAYER) if p != player]
    if end in COMMON and board[enemies, end].sum() > 0:
        enemy = np.nonzero(board[:, end])[0].item()
        board[enemy, end] -= 1
        board[enemy, 0] += 1
    board[player, start] -= 1
    board[player, end] += 1


def get_legal_moves(board, player, dice):
    if dice == 0:
        return []

    enemies = [p for p in range(N_PLAYER) if p != player]
    inds = np.nonzero(board[player, :-1])[-1].tolist()

    moves = []
    for start in inds:
        end = start + dice
        if start > max(COMMON) and end > N_BOARD + 1:
            # Player needs exact throw for last two squares
            continue
        end = min(end, N_BOARD + 1)
        if end < N_BOARD + 1 and board[[player], end].sum() > 0:
            # Player has a piece there
            continue
        if end in SAFE and board[enemies, end].sum() > 0:
            # Ennemy on safe rosette
            continue
        moves.append((start, end))

    return moves


def determine_winner(board):
    # Empty if no winners
    return np.nonzero(board[:, -1] == N_PIECE)[0].tolist()


def throw():
    return np.random.randint(0, 2, size=4, dtype=np.uint8).sum().item()
