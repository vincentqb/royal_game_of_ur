import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import play_one
import torch
from play_one import N_BOARD, N_PLAYER, play
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


def compare_play_wrapper(play, selected):
    winner = play([getattr(play_one, s) for s in selected])
    return {
        **{k: v for k, v in enumerate(selected)},
        "winner_id": winner,
        "winner_name": selected[winner],
    }


def compare():
    available = ["policy_first", "policy_last", "policy_random", "policy_aggressive"]

    tasks = []
    for _ in range(2000):
        selected = random.choices(available, k=2)
        tasks.append([play, selected])

    results = list(parallel_map(compare_play_wrapper, tasks))
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


def train_play_wrapper(play, selected):
    winner, boards = play([getattr(play_one, s) for s in selected], return_boards=True)
    return {
        **{k: v for k, v in enumerate(selected)},
        "boards": boards,
        "winner_id": winner,
        "winner_name": selected[winner],
    }


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_hidden = N_PLAYER * (N_BOARD + 3)
        self.fc0 = torch.nn.Linear(n_hidden, n_hidden, dtype=torch.float16)
        self.fc1 = torch.nn.Linear(n_hidden, N_PLAYER, dtype=torch.float16)

    def forward(self, board):
        board = board.to(torch.float16)
        board = self.fc0(board)
        board = torch.nn.functional.leaky_relu(board)
        board = self.fc1(board)
        return board


def train():
    available = ["policy_first", "policy_last", "policy_random", "policy_aggressive"]

    net = Net()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    for _ in range(10):
        tasks = []
        for _ in range(100):
            selected = random.choices(available, k=2)
            tasks.append([play, selected])

        results = list(parallel_map(train_play_wrapper, tasks))
        xs, ys = zip(*[(r["boards"], r["winner_id"]) for r in results if r["winner_id"] >= 0])

        xs = np.vstack(xs)
        ys = np.vstack(ys)
        xs = torch.from_numpy(xs)
        ys = torch.from_numpy(ys)
        ys = torch.nn.functional.one_hot(ys, num_classes=N_PLAYER)

        optimizer.zero_grad()
        yh = net(xs)
        loss = torch.nn.functional.mse_loss(ys, yh)
        loss.backward()
        optimizer.step()

        print(loss.item())
    return net


if __name__ == "__main__":
    # results = compare()
    train()
