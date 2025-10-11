import random
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

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


def configure_logger(exp_dir):
    logger.remove()
    logger.add(exp_dir / "trace.log", level="TRACE", enqueue=True, serialize=True)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")

    def showwarning_to_loguru(message, category, filename, lineno, file=None, line=None):
        formatted_message = warnings.formatwarning(message, category, filename, lineno, line)
        logger.warning(formatted_message)

    warnings.showwarning = showwarning_to_loguru


class UrNet(nn.Module):
    """Neural network for policy and value prediction."""

    def __init__(self, input_size=(N_PLAYER * (N_BOARD + 2)), hidden_size=128, device=None):
        super().__init__()

        slope = 0.01
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=dtype, device=device),
            nn.LeakyReLU(slope),
            nn.Linear(hidden_size, hidden_size, dtype=dtype, device=device),
            nn.LeakyReLU(slope),
            nn.Linear(hidden_size, hidden_size // 2, dtype=dtype, device=device),
            nn.LeakyReLU(slope),
        )

        self.policy = nn.Linear(hidden_size // 2, N_BOARD + 2, dtype=dtype, device=device)

        self.value = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4, dtype=dtype, device=device),
            nn.LeakyReLU(slope),
            nn.Linear(hidden_size // 4, 1, dtype=dtype, device=device),
            nn.Tanh(),
        )

    def forward(self, x):
        features = self.shared(x)
        policy_logits = self.policy(features)
        value = self.value(features)
        return policy_logits, value


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

    board = torch.from_numpy(board.astype(np.float32).flatten()).to(device).to(dtype)

    with torch.inference_mode():
        net = net.to(device)
        policy_logits, value = net(board)
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
        move, _, _ = select_move(net, board, player, moves, device=device, training=False)
        return move

    return policy_neural


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, maxlen=50_000):
        self.buffer = deque(maxlen=maxlen)

    def append(self, board, move_probs, reward):
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
            break

        iteration += 1

    # Calculate rewards based on outcome
    training_data = []

    # Get final scores from the final board state, normalized to [-1, 1]
    # Note: experiences store standardized boards, so get from actual game board
    # final_score_margin = (board[0, -1].item() - board[1:, -1].max().item()) / N_PIECE

    # discount_rate = 1.0
    discount_rate = 0.999  # NOTE 0.999**1_000 ~= 0.37
    for i, (exp_board, exp_player, exp_probs) in enumerate(experiences):
        if winner:
            # Base win/loss reward
            if exp_player == winner[0]:
                base_reward = +1.0
            else:
                base_reward = -1.0
        else:
            # Draw or timeout
            base_reward = 0.0

        # Margin bonus/penalty
        # if exp_player == 0:
        #     margin_reward = +final_score_margin
        # else:
        #     margin_reward = -final_score_margin

        # Current score
        margin_reward = (exp_board[0, -1].item() - exp_board[1:, -1].max().item()) / N_PIECE

        # Combine: winner gets base + margin, loser gets base - margin
        alpha = 0.7
        reward = alpha * (discount_rate ** (len(experiences) - i)) * base_reward + (1.0 - alpha) * margin_reward

        # Clip to valid range
        clip = 0.999
        reward = np.clip(reward, -clip, clip)

        training_data.append((exp_board, exp_probs, reward))

    return training_data


def train_batch(net, optimizer, batch, *, device):
    """Train on a batch of experiences."""
    if not batch:
        return 0.0, 0.0, 0.0

    boards_np = np.stack([exp["board"].astype(np.float32) for exp in batch])
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
    alpha = 0.5
    loss = alpha * policy_loss + (1 - alpha) * value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item(), rewards.abs().sum().item()


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


def evaluate_models(policies, num_games=50):
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

    tasks = []
    for _ in range(num_games):
        selected = random.choices(policies, k=2)
        tasks.append([selected])

    results = list(parallel_map(compare_play_wrapper, tasks))

    with pd.option_context("display.float_format", "{:.0f}".format):
        elos = compare_elo(results)
        logger.info("\nELO Ratings:\n{elos}", elos=elos)

    with pd.option_context("display.float_format", "{:.4f}".format):
        pairwise = compare_pairwise(results)
        logger.info("\nPairwise Win Rates:\n{pairwise}", pairwise=pairwise)

    return elos, pairwise


def train(
    *,
    exp_dir,
    batch_size=25,
    num_batches=500,
    num_iterations=500,
    save_interval=25,
):
    """
    Main training loop.

    Args:
        exp_dir: Path to experiment directory
        batch_size: Number of self-play games per iteration
        num_batches: Number of training batches per iteration
        num_iterations: Number of training iterations
        save_interval: Save and evaluate model every N iterations

    Returns:
        net: Trained network
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    logger.trace("Using device: {device}", device=device)

    net = UrNet(device=device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    buffer = ReplayBuffer()

    best_elo = 0.0
    best_model_path = exp_dir / "best_model.pt"

    try:
        for iteration in trange(num_iterations, ncols=0, desc="Epoch"):
            net.eval()  # Set to eval mode for inference

            temperature = max(0.5, 3.0 - 2.5 * iteration / num_iterations)
            tasks = [(net, temperature, device) for _ in range(batch_size)]
            for experiences in parallel_map(self_play_game, tasks):
                for exp_board, exp_probs, reward in experiences:
                    buffer.append(exp_board, exp_probs, reward)

            logger.trace("length of buffer: {length}", length=len(buffer))

            if len(buffer) >= batch_size:
                net.train()

                num_batches = min(num_batches, len(buffer) // batch_size)
                total_loss = 0.0
                total_policy_loss = 0.0
                total_value_loss = 0.0
                total_reward = 0.0

                for _ in range(num_batches):
                    batch = buffer.sample(batch_size)
                    loss, p_loss, v_loss, reward = train_batch(net, optimizer, batch, device=device)
                    total_loss += loss
                    total_policy_loss += p_loss
                    total_value_loss += v_loss
                    total_reward += reward

                avg_loss = total_loss / num_batches
                avg_p_loss = total_policy_loss / num_batches
                avg_v_loss = total_value_loss / num_batches
                avg_reward = total_reward / num_batches  # Mean absolute reward

                logger.info(
                    "Loss: {loss:.4f} - Policy: {p_loss:.4f} - Value: {v_loss:.4f} - Reward: {reward:.4f}",
                    loss=avg_loss,
                    p_loss=avg_p_loss,
                    v_loss=avg_v_loss,
                    reward=avg_reward,
                    iteration=iteration,
                )

            # Save checkpoint and evaluate
            if (iteration + 1) % save_interval == 0:
                checkpoint_path = exp_dir / f"checkpoint_{iteration + 1:05d}.pt"

                torch.save(
                    {
                        "iteration": iteration,
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_path,
                )
                logger.debug("Checkpoint saved to {path}", path=checkpoint_path)

                # Now evaluate the saved model
                elos, pairwise = evaluate_models([str(checkpoint_path), "policy_random", "policy_aggressive"])
                neural_elo = elos[str(checkpoint_path)]

                # Update checkpoint with ELO metadata
                is_best = neural_elo > best_elo
                # checkpoint = torch.load(checkpoint_path, weights_only=False)
                # checkpoint.update(
                #     {
                #         "elo": neural_elo,
                #         "best_elo": neural_elo if is_best else best_elo,
                #         "is_best": is_best,
                #     }
                # )
                # torch.save(checkpoint, checkpoint_path)

                # Update best model symlink if this is the best
                if is_best:
                    best_elo = neural_elo
                    if best_model_path.exists() or best_model_path.is_symlink():
                        best_model_path.unlink()
                    best_model_path.symlink_to(checkpoint_path.name)
                    logger.success("New best model with ELO: {elo:.0f}", elo=neural_elo)

    except KeyboardInterrupt:
        logger.warning("Training interupted by user at iteration {iteration}", iteration=iteration)

    logger.success("Best ELO: {elo:.0f}", elo=best_elo)
    return checkpoint_path


if __name__ == "__main__":
    # Create experiment directory
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/{experiment_id}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    configure_logger(exp_dir)
    logger.info("Experiment directory: {exp_dir}", exp_dir=exp_dir)

    # Train the agent
    checkpoint_model_path = train(exp_dir=exp_dir)
    best_model_path = exp_dir / "best_model.pt"

    # Final evaluation
    available = [
        "policy_random",
        "policy_aggressive",
        "policy_first",
        "policy_last",
    ]
    if best_model_path.exists():
        available.append(str(best_model_path))
    if checkpoint_model_path.exists():
        available.append(str(checkpoint_model_path))
    evaluate_models(available, num_games=200)
