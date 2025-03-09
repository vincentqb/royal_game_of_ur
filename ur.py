import random

import numpy as np
import pandas as pd
from tqdm import trange

n_player = 2

# all start on first and end on last
n_board = 14
ind_safe = sorted([8])
ind_rosette = sorted([4, 8, 14])
ind_common = sorted(range(5, 13))
n_pieces = 7


def create_board():
    board = np.zeros((n_player, n_board + 2))
    board[:, 0] = n_pieces
    return board


def throw():
    return np.random.randint(0, 2, size=4).sum().item()


def get_legal_moves(board, player, dice):
    if dice == 0:
        return []

    enemies = [p for p in range(n_player) if p != player]
    inds = np.nonzero(board[player, :-1])[-1].tolist()

    moves = []
    for start in inds:
        end = start + dice
        if start > max(ind_common) and end > n_board + 1:
            # Player needs exact throw for last two squares
            continue
        end = min(end, n_board + 1)
        if end < n_board + 1 and board[[player], end].sum() > 0:
            # Player has a piece there
            continue
        if end in ind_safe and board[enemies, end].sum() > 0:
            # Ennemy on safe rosette
            continue
        moves.append((start, end))

    return moves


def determine_winner(board):
    # Empty if no winners
    return np.nonzero(board[:, -1] == n_pieces)[0].tolist()


def execute_move(board, player, start, end):
    enemies = [p for p in range(n_player) if p != player]
    if end in ind_common and board[enemies, end].sum() > 0:
        enemy = np.nonzero(board[:, end])[0].item()
        board[enemy, end] -= 1
        board[enemy, 0] += 1
    board[player, start] -= 1
    board[player, end] += 1
    return board


def print_board(board):
    # separator = " | "
    separator = ""
    rows = ["" for _ in range(n_player + 1)]
    for i in ind_common:
        player = np.nonzero(board[:, i])[0]
        if len(player) > 0:
            rows[-1] += f"{int(player[0].item())}"
        elif i in ind_rosette:
            rows[-1] += "R"
        else:
            rows[-1] += "*"
    for player in range(n_player):
        inds = list(range(max(ind_common) + 1, board.shape[-1])) + list(range(min(ind_common)))
        for i in inds:
            if i in [0, n_board + 1]:
                rows[player] += f"{int(board[player, i])}"
            elif board[player, i] > 0:
                rows[player] += f"{int(player)}"
            elif i in ind_rosette:
                rows[player] += "R"
            else:
                rows[player] += "*"
        rows[player] = (" " * (len(ind_common) - len(ind_common))) + rows[player]
        rows[player] = rows[player][::-1]

    rows[-2], rows[-1] = rows[-1], rows[-2]
    print("\n".join(rows))


def get_player_move(moves):
    moves = {k: m for k, m in enumerate(moves)}
    while True:
        move = input(f"Select move {moves}: ")
        try:
            move = int(move)
            if move in range(len(moves)):
                break
        except:
            print("Invalid entry.")
    return moves[move]


def play_first(moves):
    return moves[0]


def play_last(moves):
    return moves[-1]


def play_random(moves):
    return random.choice(moves)


def play(get_move_for_player, verbose=False):
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
            move = get_move_for_player[player](moves)
            board = execute_move(board, player, *move)
            if move[-1] in ind_rosette:
                # Play again on rosettes
                continue
        elif verbose:
            print(f"Player {player} has no legal moves.")

        player = (player + 1) % n_player
        winner = determine_winner(board)
        iteration += 1

        if iteration > 1000:
            return [-1]

    return winner[0]


def compare():
    available = ["play_first", "play_last", "play_random"]

    results = []
    for _ in trange(2000, ncols=0):
        selected = random.choices(available, k=2)
        winner = play([eval(s) for s in selected])
        results.append(
            {
                **{k: v for k, v in enumerate(selected)},
                "winner_id": winner,
                "winner_name": selected[winner],
            }
        )

    results = pd.DataFrame(results, columns=list(range(n_player)) + ["winner_id", "winner_name"])

    results_winner = [
        results[results["winner_id"] == p]
        .value_counts()
        .reset_index()
        .pivot(index=0, columns=1, values="count")
        .fillna(0)
        for p in range(n_player)
    ]

    p = results_winner[0] + results_winner[1].transpose()
    print(p / (p + p.transpose()))
    return results


if __name__ == "__main__":
    # winner = play([play_last, play_last], verbose=True)
    # winner = play([get_player_move, get_player_move])
    # print(f"Player {winner} won.")

    results = compare()
