import random
from contextlib import nullcontext

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
from policies import POLICIES
from rich import print
from rich.live import Live


def play(policies, board=None, show=False):
    board = create_board() if board is None else board
    player = random.randrange(N_PLAYER)
    winner = []
    iteration = 0
    max_iterations = 1000
    experiences = []

    policies = [POLICIES[policy] for policy in policies]

    if show:
        print(f"Player {player} starts.")

    with Live(auto_refresh=False) if show else nullcontext() as visual:
        while True:
            dice = throw()
            moves = get_legal_moves(board, player, dice)

            if show:
                print(f"Player {player} threw {dice}.")

            if moves:
                std_board = standardize_state(board, player)
                move = policies[player](
                    board=board, std_board=std_board, player=player, moves=moves, visual=visual if show else None
                )
                if move == -1:
                    if show:
                        print("Players quit.")
                    break
                experience = dict(
                    board=std_board.copy(),
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
                    if show:
                        print(f"Player {player} plays again.")
                    continue

            player = (player + 1) % N_PLAYER

            iteration += 1
            if iteration > max_iterations:
                if show:
                    print("Game is too long.")
                break

    return experiences


if __name__ == "__main__":
    play(
        [
            "human",
            "urnet_00350",
        ],
        show=True,
    )
    # play(["human", "human"], show=True)
