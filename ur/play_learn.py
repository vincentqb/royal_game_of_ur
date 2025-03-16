import random

import numpy as np
import play_one
import torch
from play_many import parallel_map
from play_one import N_BOARD, N_PLAYER, play


def train_play_wrapper(play, selected):
    winner, boards = play([getattr(play_one, s) for s in selected], return_boards=True)
    return {
        **{k: v for k, v in enumerate(selected)},
        "boards": boards,
        "winner_id": winner,
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
    train()
