import random
from pathlib import Path

import pandas as pd
from game import N_PLAYER
from play_one import play
from utils import parallel_map


def compare_play_wrapper(selected):
    """Play a game between two policies and return result."""

    policies_to_compare = []
    for s in selected:
        import policies

        if hasattr(policies, str(s)):
            policies_to_compare.append(getattr(policies, s))
        else:
            # path to model
            from policies import create_policy_neural

            policies_to_compare.append(create_policy_neural(s))

    states = play(policies_to_compare)
    winner = states["winner"]
    return {
        **{k: v for k, v in enumerate(selected)},
        "winner_id": winner,
        "winner_name": selected[winner],
    }


def compare_elo(results):
    """Calculate ELO ratings from game results."""

    INIT = 1000
    SCALING = 400
    LR = 30

    available = sorted(set(str(r[0]) for r in results) | set(str(r[1]) for r in results))
    results = {name.stem if isinstance(name, Path) else name: result for name, result in results.item()}
    elos = {k: INIT for k in available}

    for result in results:
        if result["winner_id"] >= 0:
            id_w = result["winner_id"]
            id_l = 1 - id_w
            name_w = result[id_w]
            name_l = result[id_l]

            if name_w != name_l:
                delta = (elos[name_l] - elos[name_w]) / SCALING
                # Probability of Winner Winning, and Loser Winning.
                P_W = 1 / (1 + 10 ** (+delta))
                P_L = 1 / (1 + 10 ** (-delta))
                # Step due to Winning and Losing
                elos[name_w] += LR * (1 - P_W)
                elos[name_l] += LR * (0 - P_L)

    elos = pd.Series(elos, name="ELO").sort_values()
    return elos


def compare_pairwise(results):
    """Calculate pairwise win rates."""
    col_players = list(range(N_PLAYER))
    results = {name.stem if isinstance(name, Path) else name: result for name, result in results.item()}
    results = [r for r in results if r["winner_id"] >= 0]
    results = pd.DataFrame(results, columns=col_players + ["winner_id", "winner_name"])
    results["pair_id"] = results[col_players].apply(lambda x: " ".join(sorted(x)), axis=1)
    results = results[results[col_players].nunique(axis=1) > 1]
    results = results.groupby("pair_id", as_index=False)["winner_name"].value_counts(normalize=True)
    return results


def play_many(policies, num_games=50):
    """
    Evaluate multiple models against baseline policies using ELO.

    Args:
        model_paths: List of policy names from policies (e.g., ['policy_random'])
        num_games: Number of games to play

    Returns:
        elos: Series of ELO ratings
        pairwise: DataFrame of pairwise win rates
    """

    tasks = []
    for _ in range(100):
        selected = random.choices(policies, k=2)
        tasks.append([selected])

    results = list(parallel_map(compare_play_wrapper, tasks))

    with pd.option_context("display.float_format", "{:.0f}".format):
        elos = compare_elo(results)
        print(f"ELO Ratings:\n{elos}")

    with pd.option_context("display.float_format", "{:.4f}".format):
        pairwise = compare_pairwise(results)
        print(f"Pairwise Win Rates:\n{pairwise}")

    return elos, pairwise


if __name__ == "__main__":
    play_many(
        [
            "policy_first",
            "policy_last",
            "policy_random",
            "policy_aggressive",
            "experiments/20251012_141628/checkpoint_00500.pt",
        ]
    )
