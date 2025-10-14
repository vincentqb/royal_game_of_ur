import curses
import random

import numpy as np
import torch
import torch.nn as nn
from game import COMMON, N_BOARD, N_PLAYER, ROSETTE
from utils import dtype


def policy_human(*, board, player, moves, visual, **_):
    move_index = 0
    while True:
        visual.show_pieces(board, moves[move_index][0], player)

        # Get user input
        key = visual.screen.getch()
        if key == ord("q"):
            return -1
        elif key == 9:  # Tab key to cycle through pieces with legal moves
            move_index = (move_index + 1) % len(moves)
        elif key == curses.KEY_LEFT:
            move_index = (move_index - 1) % len(moves)
        elif key == curses.KEY_RIGHT:
            move_index = (move_index + 1) % len(moves)
        elif key == 10:  # Enter key to confirm move
            return moves[int(move_index)]


def policy_first(*, moves, **_):
    return moves[0]


def policy_last(*, moves, **_):
    return moves[-1]


def policy_random(*, moves, **_):
    return random.choice(moves)


def policy_aggressive(*, board, player, moves, **_):
    enemies = [p for p in range(N_PLAYER) if p != player]
    for move in moves[::-1]:
        if move[-1] in COMMON and board[enemies, move[-1]].sum() > 0:
            return move
        if move[-1] in ROSETTE:
            return move
    return move


class UrNet(nn.Module):
    """Neural network for policy and value prediction."""

    def __init__(self, input_size=(N_PLAYER * (N_BOARD + 2)), hidden_size=128, device=None, return_value=False):
        super().__init__()

        self.shared = nn.ModuleList(
            [
                nn.Linear(input_size, hidden_size, dtype=dtype, device=device),
                nn.Linear(hidden_size, hidden_size, dtype=dtype, device=device),
                nn.Linear(hidden_size, hidden_size // 2, dtype=dtype, device=device),
            ]
        )

        self.policy = nn.Linear(hidden_size // 2, N_BOARD + 2, dtype=dtype, device=device)

        self.return_value = return_value
        if self.return_value:
            self.value = nn.ModuleList(
                [
                    nn.Linear(hidden_size // 2, hidden_size // 4, dtype=dtype, device=device),
                ]
            )
            self.value_final = nn.Linear(hidden_size // 4, 1, dtype=dtype, device=device)

    def forward(self, x):
        if self.training:

            def activation(x):
                return torch.nn.functional.leaky_relu(x, negative_slope=0.01)

        else:
            activation = torch.nn.functional.relu

        for layer in self.shared:
            x = layer(x)
            x = activation(x)

        policy_logits = self.policy(x)

        value = None

        if self.return_value:
            for layer in self.value:
                x = layer(x)
                x = activation(x)
            value = self.value_final(x)
            value = torch.nn.functional.tanh(value)

        return policy_logits, value


def load_model(model_path, device):
    """Load model from checkpoint."""
    net = UrNet(device=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        net.load_state_dict(checkpoint["model_state_dict"])
    else:
        net.load_state_dict(checkpoint)

    net.eval()
    return net


def get_move_mask(moves, *, device):
    """
    Create mask for legal moves.

    Returns:
        mask: Tensor with 0 for legal starting positions, -inf for illegal
        move_map: Dict mapping board positions to actual moves
    """
    mask = torch.full((N_BOARD + 2,), float("-inf"), requires_grad=False, dtype=dtype, device=device)
    move_map = {}

    for start, end in moves:
        mask[start] = 0.0
        move_map[start] = (start, end)

    return mask, move_map


def select_move(net, board, player, moves, *, device, temperature=1.0, training=True):
    """
    Select move using policy network with dice-based masking.

    Args:
        net: Neural network
        board: Current board state (numpy array)
        player: Current player
        moves: List of legal (start, end) moves
        device: torch device
        temperature: Exploration parameter
        training: If True, sample; if False, take argmax

    Returns:
        move: Selected (start, end) move
        probs: Action probabilities
    """
    if not moves:
        return None, None

    with torch.inference_mode():
        board = torch.from_numpy(board.astype(np.float32).flatten()).to(device).to(dtype)

        net = net.to(device)
        policy_logits, _ = net(board)
        policy_logits = policy_logits.squeeze(0)
        probs = torch.softmax(policy_logits / temperature, dim=0)

        mask, move_map = get_move_mask(moves, device=device)
        masked_logits = policy_logits + mask
        masked_probs = torch.softmax(masked_logits / temperature, dim=0)

        if training:
            action_idx = torch.multinomial(masked_probs, 1).item()
        else:
            action_idx = torch.argmax(masked_probs).item()

        move = move_map.get(action_idx)
        if move is None:
            move = random.choice(moves)

        return move, probs


def create_policy_neural(model_path):
    """
    Create a policy_neural function for a specific model.

    This creates a policy function that can be used with play_one.play()
    and can be pickled for multiprocessing.

    Args:
        model_path: Path to saved model checkpoint

    Returns:
        Policy function compatible with play_one interface
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    net = load_model(model_path, device)

    def policy_neural(board, player, moves, **kwargs):
        move, _ = select_move(net, board, player, moves, device=device, training=False)
        return move

    return policy_neural
