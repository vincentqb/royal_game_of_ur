import random

import numpy as np
import pandas as pd
import play_one
import torch
from play_many import parallel_map
from play_one import N_BOARD, N_PLAYER, execute_move, flatten, play


def train_play_wrapper(*policies):
    names, funcs = zip(*policies)
    winner, boards = play(funcs, return_boards=True)
    winners = np.repeat(winner, 1 if boards is None else boards.shape[0])
    return {
        **{k: v for k, v in enumerate(names)},
        "boards": boards,
        "winners": winners,
        "winner_id": winner,
        "winner_name": names[winner],
    }


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_hidden = N_PLAYER * (N_BOARD + 2) + 1
        self.fc0 = torch.nn.Linear(n_hidden, n_hidden, dtype=torch.float32)
        self.fc1 = torch.nn.Linear(n_hidden, N_PLAYER, dtype=torch.float32)

    def forward(self, board):
        board = self.fc0(board)
        board = torch.nn.functional.leaky_relu(board)
        board = self.fc1(board)
        board = torch.nn.functional.tanh(board)
        return board

    def policy_model(self, *, board, player, moves, **_):
        # with torch.no_grad():
        #     scores = []
        #     for move in moves:
        #         board_ = board.copy()
        #         execute_move(board_, player, *move)
        #         board_ = flatten(player, board_)
        #         board_ = torch.from_numpy(board_).to(torch.float32)
        #         score = self.forward(board_)
        #         scores.append(score)
        # move = scores.index(max(scores))  # argmax
        with torch.no_grad():
            boards = []
            for move in moves:
                board_ = board.copy()
                execute_move(board_, player, *move)
                board_ = flatten(board_, player)
                boards.append(board_)
            boards = np.vstack(boards, dtype=np.float32)
            boards = torch.from_numpy(boards)
            scores = self.forward(boards)
            scores = scores[:, player]
            move = torch.argmax(scores)
        return moves[move]


def compare(results):
    results = pd.DataFrame(results, columns=list(range(N_PLAYER)) + ["winner_id", "winner_name"])

    results_winner = [
        results[results["winner_id"] == p]
        .value_counts()
        .reset_index()
        .pivot(index=0, columns=1, values="count")
        .fillna(0)
        for p in range(N_PLAYER)
    ]

    p = results_winner[0] + results_winner[1].transpose()
    print(p / (p + p.transpose()))
    return results


def train():
    net = Net()
    POLICIES = {
        "first": play_one.policy_first,
        "last": play_one.policy_last,
        "random": play_one.policy_random,
        "aggressive": play_one.policy_aggressive,
        "model": net.policy_model,
    }

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    for _ in range(100):
        tasks = []
        for _ in range(1000):
            selected = random.choices(list(POLICIES), k=N_PLAYER)
            tasks.append([(select, POLICIES[select]) for select in selected])

        results = list(parallel_map(train_play_wrapper, tasks))
        compare(results)

        xs, ys = zip(*[(r["boards"], r["winners"]) for r in results if r["winner_id"] >= 0])

        xs = np.vstack(xs, dtype=np.float32)
        ys = np.concat(ys)
        xs = torch.from_numpy(xs)
        ys = torch.from_numpy(ys)
        ys = torch.nn.functional.one_hot(ys, num_classes=N_PLAYER)
        ys = 2 * (ys - 1/2)
        ys = ys.to(torch.float32)

        optimizer.zero_grad()
        yh = net(xs)
        loss = torch.nn.functional.mse_loss(ys, yh)
        loss.backward()
        optimizer.step()

        print(loss.item())
    return net


if __name__ == "__main__":
    train()
