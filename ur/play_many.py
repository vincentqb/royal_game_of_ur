import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import play_one
from play_one import N_PLAYER, play
from tqdm import tqdm


def parallel_map(func, args, max_workers=None):
    """Applies func to items in args (list of tuples), preserving order."""
    if max_workers is None or max_workers >= 0:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func, *arg) for arg in args]
            for future in tqdm(as_completed(futures), total=len(futures), ncols=0):
                yield future.result()
    else:
        for arg in tqdm(args, ncols=0):
            yield func(*arg)


def compare_play_wrapper(selected):
    states = play([getattr(play_one, s) for s in selected])
    winner = states["winner"]
    return {
        **{k: v for k, v in enumerate(selected)},
        "winner_id": winner,
        "winner_name": selected[winner],
    }


def compute_elo(results):
    INIT = 1000
    SCALING = 400
    LR = 30

    available = sorted(set(r[0] for r in results) | set(r[1] for r in results))
    elos = {k: INIT for k in available}

    for result in results:
        if result["winner_id"] >= 0:
            id_w = result["winner_id"]
            id_l = 1 - id_w
            name_w = result[id_w]
            name_l = result[id_l]

            if name_w != name_l:
                step = 1 - 1 / (1 + 10 ** (elos[name_l] - elos[name_w]) / SCALING)
                elos[name_w] += LR * step
                elos[name_l] -= LR * step
    return elos


def compare():
    available = ["policy_first", "policy_last", "policy_random", "policy_aggressive"]

    tasks = []
    for _ in range(100):
        selected = random.choices(available, k=2)
        tasks.append([selected])

    results = list(parallel_map(compare_play_wrapper, tasks))

    elos = compute_elo(results)
    print(pd.Series(elos, name="ELO").sort_values().astype(int))

    results = pd.DataFrame(results, columns=list(range(N_PLAYER)) + ["winner_id", "winner_name"])

    results_draw = (
        results[results["winner_id"] == -1]
        .value_counts()
        .reset_index()
        .pivot(index=0, columns=1, values="count")
        .fillna(0)
    )
    print("Draws")
    print(results_draw)

    results_winner = [
        results[results["winner_id"] == p]
        .value_counts()
        .reset_index()
        .pivot(index=0, columns=1, values="count")
        .fillna(0)
        for p in range(N_PLAYER)
    ]

    p = results_winner[0] + results_winner[1].transpose()
    print("Winner vs Loser")
    print(p / (p + p.transpose()))
    return results


if __name__ == "__main__":
    results = compare()
