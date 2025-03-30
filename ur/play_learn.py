import random

import numpy as np
import pandas as pd
import play_one
import torch
from play_many import parallel_map
from play_one import N_BOARD, N_PLAYER, execute_move, play, standardize_state


def train_play_wrapper(*policies):
    names, funcs = zip(*policies)
    states = play(funcs)
    states["boards"] = np.vstack(
        [states["boards"], states["boards"][-1].reshape(1, *states["boards"].shape[1:])]
    )  # Repeat final state
    states["players"] = np.append(states["players"], states["players"][-1])  # Repeat final state
    winners = np.repeat(states["winner"], 1 if states["boards"] is None else states["boards"].shape[0])
    return {
        **{k: v for k, v in enumerate(names)},
        "boards": states["boards"],
        "players": states["players"],
        "winners": winners,
        "winner_id": states["winner"],
        "winner_name": names[states["winner"]],
    }


class ValueNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_board_combined = N_PLAYER * (N_BOARD + 2)
        n_hidden = 64
        self.layer_base0 = torch.nn.Linear(n_board_combined, n_hidden, dtype=torch.float32)
        self.layer_base1 = torch.nn.Linear(n_hidden, n_hidden, dtype=torch.float32)
        self.layer_value0 = torch.nn.Linear(n_hidden, 1, dtype=torch.float32)
        # self.layer_policy0 = torch.nn.Linear(n_hidden, N_BOARD, dtype=torch.float32)

    def forward(self, board):
        board = board.reshape(board.shape[0], -1)
        board = self.layer_base0(board)
        board = torch.nn.functional.leaky_relu(board)
        board = self.layer_base1(board)
        board = torch.nn.functional.leaky_relu(board)

        value = board
        value = self.layer_value0(value)
        value = value.squeeze(-1)
        value = torch.nn.functional.sigmoid(value)
        assert value.shape == (board.shape[0],), f"{value.shape=}"

        # policy = board
        # policy = self.layer_policy0(policy)
        # # policy = torch.nn.functional.log_softmax(policy, dim=-1, dtype=torch.float32)
        # assert policy.shape == ( board.shape[0], N_BOARD,), f"{policy.shape=}")

        return value

    def pick_move(self, *, board, player, moves, **_):
        if len(moves) == 1:
            return moves[0]

        boards = []
        for move in moves:
            board_ = board.copy()
            execute_move(board_, player, *move)
            board_ = standardize_state(board_, player)
            # TODO play other players assuming best move
            # TODO replay on rosetta
            boards.append(board_)
        boards = np.stack(boards, dtype=np.float32)
        boards = torch.from_numpy(boards)
        values = self.forward(boards)
        values = 1.0 - values  # Want to take value minimizing other players

        # breakpoint()
        # assert values.shape == len(moves), f"{(values.shape,len(moves))=}"

        gamma = 0.01
        A = len(moves)

        # values = values[..., player]
        # values = (values + 1) / 2  # map (-1,+1) to (0,+1)
        yb, ind = values.min(dim=0, keepdim=True)
        mask = np.zeros(values.shape, dtype=bool)
        mask[ind] = True

        probs = yb / (A * yb + gamma * (values - yb))
        summed = probs[~mask].sum(dim=0)
        probs[mask] = 1 - summed
        move = torch.multinomial(probs, 1).item()

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


def get_one_hot(targets, n_classes):
    res = np.eye(n_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [n_classes])


def train():
    net = ValueNet()
    POLICIES = {
        "first": play_one.policy_first,
        "last": play_one.policy_last,
        "random": play_one.policy_random,
        "aggressive": play_one.policy_aggressive,
        "model": net.pick_move,
    }

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    net.train()
    for _ in range(100):
        tasks = []
        for _ in range(10):
            selected = random.choices(list(POLICIES), k=N_PLAYER)
            tasks.append([(select, POLICIES[select]) for select in selected])

        results = list(parallel_map(train_play_wrapper, tasks, max_workers=-1))
        compare(results)

        boards, winners, players = zip(
            *[
                (r["boards"], r["winners"], r["players"])
                for r in results
                if r["winner_id"] >= 0 and r["boards"] is not None
            ]
        )

        boards = np.vstack(boards, dtype=np.float32)
        boards = torch.from_numpy(boards)

        players = np.concatenate(players)
        # players = get_one_hot(players, N_PLAYER).reshape(players.shape[0], N_PLAYER, 1)

        winners = np.concat(winners)
        # winners = get_one_hot(winners, N_PLAYER)
        winners = winners == players
        winners = torch.from_numpy(winners).to(torch.float32)
        winners = 2 * (winners - 1 / 2)

        optimizer.zero_grad()
        values = net(boards)
        # policy, values = net(boards)

        # boards_sliced = boards[..., 1:-1]
        # boards_sliced = np.take_along_axis(boards_sliced, players[:-1].reshape(1, players.shape[0] - 1, 1), axis=1)
        # boards_sliced = boards_sliced[players[:-1, ...]]
        # boards_sliced = np.take_along_axis(boards_sliced, players, axis=1)
        # loss_policy = torch.nn.functional.cross_entropy(policy[:-1, ...], boards_sliced)
        loss_policy = torch.tensor(0.0)

        loss_value = torch.nn.functional.mse_loss(values, winners)

        loss = loss_policy + loss_value
        loss.backward()
        optimizer.step()

        print(loss_policy.item(), loss_value.item())
    return net


if __name__ == "__main__":
    train()
