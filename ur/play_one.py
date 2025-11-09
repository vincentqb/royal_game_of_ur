import curses
from contextlib import nullcontext
from time import sleep

from game import (
    N_PLAYER,
    ROSETTE,
    create_board,
    determine_winner,
    execute_move,
    get_legal_moves,
    standardize_state,
    throw,
)
from policies import policy_human
from rich import print
from rich.live import Live


def play(policies, board=None, show=False):
    player = 0  # Starting player
    board = create_board() if board is None else board
    winner = []
    iteration = 0

    experiences = []

    with Live(auto_refresh=False) if show else nullcontext() as visual:
        while True:
            dice = throw()
            moves = get_legal_moves(board, player, dice)

            if show:
                print(f"Player {player} threw {dice}.")

            if moves:
                move = policies[player](board=board, player=player, moves=moves, visual=visual if show else None)
                if move == -1:
                    if show:
                        print("Players quit.")
                    break
                experience = dict(
                    board=standardize_state(board, player).copy(),
                    player=player,
                    dice=dice,
                    start=move[0],
                    end=move[1],
                    winner=-1,
                    reward=0.0,
                )
                experiences.append(experience)
                execute_move(board, player, *move)

                winner = determine_winner(board)
                if winner:
                    assert len(winner) == 1
                    if show:
                        print(f"Player {player} won.")
                    for experience in experiences:
                        experience["winner"] = winner[0]
                        experience["reward"] = 1.0 if experience["player"] == winner[0] else -1.0
                    break
                if move[-1] in ROSETTE:
                    # Play again on rosettes
                    continue

            if winner:
                break

            player = (player + 1) % N_PLAYER

            iteration += 1
            if iteration > 1000:
                if show:
                    print("Game is too long.")
                break

    return experiences


def play_human():
    play([policy_human, policy_human], show=True)
    sleep(1)


if __name__ == "__main__":
    # play([policy_random, policy_random])
    play_human()
