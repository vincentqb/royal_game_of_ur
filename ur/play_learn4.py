import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from play_one import (
    N_BOARD,
    N_PIECE,
    N_PLAYER,
    create_board,
    determine_winner,
    execute_move,
    get_legal_moves,
    play,
    standardize_state,
    throw,
)
from tqdm import tqdm, trange

dtype = torch.float32


def configure_logger():
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")


class UrNet(nn.Module):
    """Neural network for policy and value prediction."""

    def __init__(self, input_size=(N_PLAYER * (N_BOARD + 2)), hidden_size=128, device=None):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=dtype, device=device),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, dtype=dtype, device=device),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2, dtype=dtype, device=device),
            nn.ReLU(),
        )

        self.policy = nn.Linear(hidden_size // 2, N_BOARD + 2, dtype=dtype, device=device)

        self.value = nn.Sequential(
            nn.Linear(hidden_size // 2, 32, dtype=dtype, device=device),
            nn.ReLU(),
            nn.Linear(32, 1, dtype=dtype, device=device),
            nn.Tanh(),
        )

    def forward(self, x):
        features = self.shared(x)
        policy_logits = self.policy(features)
        value = self.value(features)
        return policy_logits, value


def board_to_tensor(board, *, device):
    """Convert board state to normalized tensor."""
    board_norm = board.astype(np.float32) / N_PIECE
    return torch.from_numpy(board_norm.flatten()).to(device).to(dtype)


def get_move_mask(moves, *, device):
    """
    Create mask for legal moves.

    Returns:
        mask: Tensor with 0 for legal starting positions, -inf for illegal
        move_map: Dict mapping board positions to actual moves
    """
    mask = torch.full((N_BOARD + 2,), float("-inf"), dtype=dtype, device=device)
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
        value: Estimated value of position
        probs: Action probabilities
    """
    if not moves:
        return None, 0.0, None

    board_tensor = board_to_tensor(board, device=device).unsqueeze(0)

    with torch.inference_mode():
        net = net.to(device)
        policy_logits, value = net(board_tensor)
        policy_logits = policy_logits.squeeze(0)

        mask, move_map = get_move_mask(moves, device=device)
        masked_logits = policy_logits + mask

        probs = torch.softmax(masked_logits / temperature, dim=0)

        if training:
            action_idx = torch.multinomial(probs, 1).item()
        else:
            action_idx = torch.argmax(probs).item()

        move = move_map.get(action_idx)
        if move is None:
            move = random.choice(moves)

        return move, value.item(), probs


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UrNet(device=device)
    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        net.load_state_dict(checkpoint["model_state_dict"])
    else:
        net.load_state_dict(checkpoint)

    net.eval()

    def policy_neural(board, player, moves, **kwargs):
        move, _, _ = select_move(net, board, player, moves, device=device, training=False)
        return move

    return policy_neural


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)

    def add(self, board, move_probs, reward):
        """Add experience to buffer."""
        self.buffer.append({"board": board, "move_probs": move_probs, "reward": reward})

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def self_play_game(net, temperature, device):
    """
    Play one game against itself using inference mode.

    Args:
        net: Neural network (shared, read-only)
        device: torch device
        temperature: Exploration temperature

    Returns:
        experiences: List of (board, move_probs, reward) tuples
    """
    board = create_board()
    player = 0
    experiences = []
    iteration = 0
    max_iterations = 1000

    net = net.to(device)

    while iteration < max_iterations:
        dice = throw()
        moves = get_legal_moves(board, player, dice)

        if moves:
            std_board = standardize_state(board, player)
            move, value, probs = select_move(
                net, std_board, player, moves, device=device, temperature=temperature, training=True
            )

            if move:
                # Convert probs to numpy immediately to avoid keeping GPU tensors
                experiences.append((std_board.copy(), player, probs.cpu().numpy()))
                execute_move(board, player, *move)

                if move[-1] not in [4, 8, 14]:
                    # Not a rosette
                    player = (player + 1) % N_PLAYER
            else:
                player = (player + 1) % N_PLAYER
        else:
            player = (player + 1) % N_PLAYER

        winner = determine_winner(board)
        if winner:
            training_data = []
            for exp_board, exp_player, exp_probs in experiences:
                # TODO add a small contribution corresponding to point difference
                # e.g. + (saved pieces - other player's saved pieces) / (number of pieces)
                reward = 1.0 if exp_player == winner[0] else -1.0
                training_data.append((exp_board, exp_probs, reward))

            return training_data

        iteration += 1

    return []


def train_batch(net, optimizer, batch, *, device):
    """Train on a batch of experiences."""
    if not batch:
        return 0.0, 0.0, 0.0

    boards_np = np.stack([exp["board"].astype(np.float32) / N_PIECE for exp in batch])
    boards = torch.from_numpy(boards_np.reshape(len(batch), -1)).to(device).to(dtype)

    move_probs_np = np.stack([exp["move_probs"] for exp in batch])
    target_probs = torch.from_numpy(move_probs_np).to(device).to(dtype)

    rewards = torch.tensor([exp["reward"] for exp in batch], dtype=dtype, device=device)

    policy_logits, values = net(boards)
    values = values.squeeze()

    # Policy loss: KL divergence
    log_probs = torch.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_probs * log_probs).sum(dim=1).mean()

    # Value loss: MSE
    value_loss = ((values - rewards) ** 2).mean()

    # Total loss
    alpha = 0.9
    loss = alpha * policy_loss + (1 - alpha) * value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item()


def parallel_map(func, args, *, max_workers=16, use_threads=True):
    """
    Applies func to items in args (list of tuples), preserving order.

    Args:
        func: Function to apply
        args: List of tuples of arguments
        max_workers: Number of workers (None for auto)
        use_threads: If True, use ThreadPoolExecutor; else ProcessPoolExecutor
    """
    if max_workers is None or max_workers >= 0:
        ExecutorClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        with ExecutorClass(max_workers=max_workers) as executor:
            futures = [executor.submit(func, *arg) for arg in args]
            for future in tqdm(as_completed(futures), total=len(futures), ncols=0):
                yield future.result()
    else:
        for arg in tqdm(args, ncols=0):
            yield func(*arg)


def compare_play_wrapper(selected):
    """Play a game between two policies and return result."""
    import play_one

    policies = []
    for s in selected:
        if hasattr(play_one, s):
            policies.append(getattr(play_one, s))
        else:
            # Assume it's a path to a model
            policies.append(create_policy_neural(s))

    states = play(policies)
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

    available = sorted(set(r[0] for r in results) | set(r[1] for r in results))
    elos = {k: INIT for k in available}

    for result in results:
        if result["winner_id"] >= 0:
            id_w = result["winner_id"]
            id_l = 1 - id_w
            name_w = result[id_w]
            name_l = result[id_l]

            if name_w != name_l:
                delta = (elos[name_l] - elos[name_w]) / SCALING
                P_W = 1 / (1 + 10 ** (+delta))
                P_L = 1 / (1 + 10 ** (-delta))
                elos[name_w] += LR * (1 - P_W)
                elos[name_l] += LR * (0 - P_L)

    elos = pd.Series(elos, name="ELO").sort_values()
    return elos


def compare_pairwise(results):
    """Calculate pairwise win rates."""
    col_players = list(range(N_PLAYER))
    results = [r for r in results if r["winner_id"] >= 0]
    results = pd.DataFrame(results, columns=col_players + ["winner_id", "winner_name"])
    results["pair_id"] = results[col_players].apply(lambda x: " ".join(sorted(x)), axis=1)
    results = results[results[col_players].nunique(axis=1) > 1]
    results = results.groupby("pair_id", as_index=False)["winner_name"].value_counts(normalize=True)
    return results


def evaluate_models(model_paths, baseline_policies, num_games=50):
    """
    Evaluate multiple models against baseline policies using ELO.

    Args:
        model_paths: List of paths to model checkpoints or dict {name: path}
        baseline_policies: List of policy names from play_one (e.g., ['policy_random'])
        num_games: Number of games to play

    Returns:
        elos: Series of ELO ratings
        pairwise: DataFrame of pairwise win rates
    """
    if isinstance(model_paths, dict):
        available = list(model_paths.values()) + baseline_policies
    else:
        available = model_paths + baseline_policies

    tasks = []
    for _ in range(num_games):
        selected = random.choices(available, k=2)
        tasks.append([selected])

    results = list(parallel_map(compare_play_wrapper, tasks))

    with pd.option_context("display.float_format", "{:.0f}".format):
        elos = compare_elo(results)
        logger.info(f"\nELO Ratings:\n{elos}")

    with pd.option_context("display.float_format", "{:.4f}".format):
        pairwise = compare_pairwise(results)
        logger.info(f"\nPairwise Win Rates:\n{pairwise}")

    return elos, pairwise


def train(
    batch_size=50,
    num_batches=100,
    num_iterations=500,
    eval_interval=10,
    save_interval=50,
):
    """
    Main training loop.

    Args:
        batch_size: Training batch size
        num_iterations: Number of training iterations
        eval_interval: Evaluate every N iterations
        save_interval: Save model every N iterations

    Returns:
        net: Trained network
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.trace(f"{device=}")

    net = UrNet(device=device)
    optimizer = optim.Adam(net.parameters(), lr=0.003)
    buffer = ReplayBuffer()

    best_elo = 0.0

    for iteration in trange(num_iterations, ncols=0, desc="Epoch"):
        # Self-play phase (parallel with threading)
        temperature = max(0.5, 2.0 - iteration / num_iterations / 2.0)

        net.eval()  # Set to eval mode for inference

        tasks = [(net, temperature, device) for _ in range(batch_size)]
        for experiences in parallel_map(self_play_game, tasks):
            for exp_board, exp_probs, reward in experiences:
                buffer.add(exp_board, exp_probs, reward)

        # Training phase
        if len(buffer) >= batch_size:
            net.train()  # Set to train mode

            num_batches = min(num_batches, len(buffer) // batch_size)
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0

            for _ in range(num_batches):
                batch = buffer.sample(batch_size)
                loss, p_loss, v_loss = train_batch(net, optimizer, batch, device=device)
                total_loss += loss
                total_policy_loss += p_loss
                total_value_loss += v_loss

            avg_loss = total_loss / num_batches
            avg_p_loss = total_policy_loss / num_batches
            avg_v_loss = total_value_loss / num_batches

            logger.info(f"Loss: {avg_loss:.4f} - Policy: {avg_p_loss:.4f} - Value: {avg_v_loss:.4f}")

        # Evaluation phase
        if (iteration + 1) % eval_interval == 0:
            eval_path = "ur_eval_temp.pt"
            torch.save(net.state_dict(), eval_path)

            elos, pairwise = evaluate_models([eval_path], ["policy_random", "policy_aggressive"])
            neural_elo = elos[eval_path]

            if neural_elo > best_elo:
                filename = "ur_best_model.pt"
                torch.save(net.state_dict(), filename)
                logger.debug(f"New best model saved saved to {filename}")
                logger.success(f"New best model saved - ELO: {neural_elo:.0f}")

        # Save checkpoint
        if (iteration + 1) % save_interval == 0:
            filename = f"ur_checkpoint_{iteration + 1}.pt"
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "current_elo": neural_elo,
                    "best_elo": best_elo,
                },
                filename,
            )
            logger.debug(f"Checkpoint saved to {filename}")

    logger.success(f"Best ELO: {best_elo:.0f}")

    return net


if __name__ == "__main__":
    configure_logger()

    # Train the agent
    trained_net = train()

    # Save final model
    filename = "ur_final_model.pt"
    torch.save(trained_net.state_dict(), filename)
    logger.info(f"\nFinal model saved as '{filename}'")

    # Final evaluation
    evaluate_models(
        ["ur_final_model.pt"],
        ["policy_random", "policy_aggressive", "policy_first", "policy_last"],
        num_games=200,
    )
