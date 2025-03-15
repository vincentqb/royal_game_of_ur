import random

import numpy as np

N_PLAYER = 2
N_PIECE = 7
N_BOARD = 14
SAFE = sorted([8])
ROSETTE = sorted([4, 8, 14])
COMMON = sorted(range(5, 13))


def create_board():
    # pieces start on [0] and end on [-1]
    board = np.zeros((N_PLAYER, N_BOARD + 2))
    board[:, 0] = N_PIECE
    return board


def throw():
    return np.random.randint(0, 2, size=4).sum().item()


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


def execute_move(board, player, start, end):
    enemies = [p for p in range(N_PLAYER) if p != player]
    if end in COMMON and board[enemies, end].sum() > 0:
        enemy = np.nonzero(board[:, end])[0].item()
        board[enemy, end] -= 1
        board[enemy, 0] += 1
    board[player, start] -= 1
    board[player, end] += 1
    return board


def print_board(board):
    rows = ["" for _ in range(N_PLAYER + 1)]
    for i in COMMON:
        player = np.nonzero(board[:, i])[0]
        if len(player) > 0:
            rows[-1] += f"{int(player[0].item())}"
        elif i in ROSETTE:
            rows[-1] += "R"
        else:
            rows[-1] += "*"
    for player in range(N_PLAYER):
        inds = list(range(max(COMMON) + 1, board.shape[-1])) + list(range(min(COMMON)))
        for i in inds:
            if i in [0, N_BOARD + 1]:
                rows[player] += f"{int(board[player, i])}"
            elif board[player, i] > 0:
                rows[player] += f"{int(player)}"
            elif i in ROSETTE:
                rows[player] += "R"
            else:
                rows[player] += "*"
        rows[player] = (" " * (len(COMMON) - len(COMMON))) + rows[player]
        rows[player] = rows[player][::-1]

    rows[-2], rows[-1] = rows[-1], rows[-2]
    print("\n".join(rows))


def policy_human(*, board, player, moves):
    moves = {k: m for k, m in enumerate(moves)}
    while True:
        move = input(f"Select move {moves} for Player {player}: ")
        try:
            move = int(move)
            if move in moves:
                break
        except:
            print("Invalid entry.")
    return moves[move]


def policy_first(*, moves, **_):
    return moves[0]


def policy_last(*, moves, **_):
    return moves[-1]


def policy_random(*, moves, **_):
    return random.choice(moves)


def policy_aggressive(*, board, player, moves):
    enemies = [p for p in range(N_PLAYER) if p != player]
    for move in moves[::-1]:
        if move[-1] in COMMON and board[enemies, move[-1]].sum() > 0:
            return move
        if move[-1] in ROSETTE:
            return move
    return move


def play(policies, verbose=False):
    player = 0  # starting player
    board = create_board()
    winner = []
    iteration = 0

    while not winner:
        if verbose:
            print_board(board)
        dice = throw()
        moves = get_legal_moves(board, player, dice)
        if verbose:
            print(f"Player {player} threw {dice}.")
        if moves:
            move = policies[player](board=board, player=player, moves=moves)
            board = execute_move(board, player, *move)
            if move[-1] in ROSETTE:
                # Play again on rosettes
                continue
        elif verbose:
            print(f"Player {player} has no legal moves.")

        player = (player + 1) % N_PLAYER
        winner = determine_winner(board)

        iteration += 1
        if iteration > 1000:
            return -1

    return winner[0]


if __name__ == "__main__":
    winner = play([policy_human, policy_human])
    print(f"Player {winner} won.")
