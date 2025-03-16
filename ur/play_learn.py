import random

import numpy as np
import play_one
import torch
from play_many import parallel_map
from play_one import N_BOARD, N_PLAYER, execute_move, play


def train_play_wrapper(play, selected):
    winner, boards = play(selected, return_boards=True)
    winner = np.repeat(winner, 1 if boards is None else boards.shape[0])
    return {
        **{k: v for k, v in enumerate(selected)},
        "boards": boards,
        "winner_id": winner,
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
        scores = []
        with torch.no_grad():
            board_ = board.copy()
            board_ = np.concat([[player], board.flatten()], dtype=np.float32)
            board_ = torch.from_numpy(board_).to(torch.float32)
            score = self.forward(board_)
            scores.append(score)
        move = scores.index(max(scores))  # argmax
        return moves[move]


def train():
    net = Net()
    available = [
        play_one.policy_first,
        play_one.policy_last,
        play_one.policy_random,
        play_one.policy_aggressive,
        net.policy_model,
    ]

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    for _ in range(10):
        tasks = []
        for _ in range(100):
            selected = random.choices(available, k=2)
            tasks.append([play, selected])

        results = list(parallel_map(train_play_wrapper, tasks))
        xs, ys = zip(*[(r["boards"], r["winner_id"]) for r in results if r["winner_id"][0] >= 0])

        xs = np.vstack(xs, dtype=np.float32)
        ys = np.concat(ys)
        xs = torch.from_numpy(xs)
        ys = torch.from_numpy(ys)
        ys = torch.nn.functional.one_hot(ys, num_classes=N_PLAYER)
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
