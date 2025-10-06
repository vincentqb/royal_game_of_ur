import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
from tqdm import tqdm


class UrNet(nn.Module):
    """Neural network for policy and value prediction."""

    def __init__(self, input_size=(N_PLAYER * (N_BOARD + 2)), hidden_size=128, device=None):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size, device=device),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, device=device),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2, device=device),
            nn.ReLU(),
        )

        self.policy = nn.Linear(hidden_size // 2, N_BOARD + 2, device=device)

        self.value = nn.Sequential(
            nn.Linear(hidden_size // 2, 32, device=device), nn.ReLU(), nn.Linear(32, 1, device=device), nn.Tanh()
        )

    def forward(self, x):
        features = self.shared(x)
        policy_logits = self.policy(features)
        value = self.value(features)
        return policy_logits, value


def board_to_tensor(board, device):
    """Convert board state to tensor (normalized)."""
    board_norm = board.astype(np.float32) / N_PIECE
    return torch.from_numpy(board_norm.flatten()).to(device)


def get_move_mask(moves, device):
    """Create mask for legal moves.

    Returns:
        mask: Tensor with 0 for legal starting positions, -inf for illegal
        move_map: Dict mapping board positions to actual moves
    """
    mask = torch.full((N_BOARD + 2,), float("-inf"), device=device)
    move_map = {}

    for start, end in moves:
        mask[start] = 0.0
        move_map[start] = (start, end)

    return mask, move_map


def select_move(net, board, player, moves, device, temperature=1.0, training=True):
    """Select move using policy network with dice-based masking.

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

    board_tensor = board_to_tensor(board, device).unsqueeze(0)

    with torch.no_grad():
        policy_logits, value = net(board_tensor)
        policy_logits = policy_logits.squeeze(0)

        mask, move_map = get_move_mask(moves, device)
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


def create_policy_neural(model_path, temperature=0.1):
    """Create a policy_neural function for a specific model.

    This creates a policy function that can be used with play_one.play()
    and can be pickled for multiprocessing.

    Args:
        model_path: Path to saved model checkpoint
        temperature: Temperature for move selection

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
        move, _, _ = select_move(net, board, player, moves, device, temperature, training=False)
        return move

    return policy_neural


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)

    def add(self, board, move_probs, reward):
        """Add experience to buffer."""
        self.buffer.append({"board": board, "move_probs": move_probs.cpu().numpy(), "reward": reward})

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def self_play_game(net, device, temperature=1.0):
    """Play one game against itself.

    Returns:
        experiences: List of (board, move_probs, reward) tuples
    """
    board = create_board()
    player = 0
    experiences = []
    iteration = 0
    max_iterations = 1000

    while iteration < max_iterations:
        dice = throw()
        moves = get_legal_moves(board, player, dice)

        if moves:
            std_board = standardize_state(board, player)
            move, value, probs = select_move(net, std_board, player, moves, device, temperature, training=True)

            if move:
                experiences.append((std_board.copy(), player, probs))
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
                reward = 1.0 if exp_player == winner[0] else -1.0
                training_data.append((exp_board, exp_probs, reward))

            return training_data

        iteration += 1

    return []


def train_batch(net, optimizer, batch, device):
    """Train on a batch of experiences."""
    if not batch:
        return 0.0, 0.0, 0.0

    boards_np = np.stack([exp["board"].astype(np.float32) / N_PIECE for exp in batch])
    boards = torch.from_numpy(boards_np.reshape(len(batch), -1)).to(device)

    move_probs_np = np.stack([exp["move_probs"] for exp in batch])
    target_probs = torch.from_numpy(move_probs_np).to(device)

    rewards = torch.tensor([exp["reward"] for exp in batch], dtype=torch.float32, device=device)

    policy_logits, values = net(boards)
    values = values.squeeze()

    # Policy loss: KL divergence
    log_probs = torch.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_probs * log_probs).sum(dim=1).mean()

    # Value loss: MSE
    value_loss = ((values - rewards) ** 2).mean()

    # Total loss
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item()


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


def evaluate_models(model_paths, baseline_policies, num_games=100, max_workers=None):
    """Evaluate multiple models against baseline policies using ELO.

    Args:
        model_paths: List of paths to model checkpoints or dict {name: path}
        baseline_policies: List of policy names from play_one (e.g., ['policy_random'])
        num_games: Number of games to play
        max_workers: Number of parallel workers (None for auto)

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

    results = list(parallel_map(compare_play_wrapper, tasks, max_workers=max_workers))

    pd.set_option("display.float_format", "{:.0f}".format)
    elos = compare_elo(results)
    print("\nELO Ratings:")
    print(elos)

    pd.set_option("display.float_format", "{:.4f}".format)
    pairwise = compare_pairwise(results)
    print("\nPairwise Win Rates:")
    print(pairwise)

    return elos, pairwise


def train(num_iterations=500, games_per_iter=50, batch_size=64, eval_interval=10, save_interval=50, lr=0.001):
    """Main training loop.

    Args:
        num_iterations: Number of training iterations
        games_per_iter: Self-play games per iteration
        batch_size: Training batch size
        eval_interval: Evaluate every N iterations
        save_interval: Save model every N iterations
        lr: Learning rate

    Returns:
        net: Trained network
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Initializing AlphaUr training...")
    print(f"Device: {device}")

    net = UrNet(device=device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    buffer = ReplayBuffer(max_size=50000)

    best_win_rate = 0.0

    for iteration in range(num_iterations):
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'=' * 60}")

        # Self-play phase
        print(f"Self-play: generating {games_per_iter} games...")
        temperature = max(0.5, 1.0 - iteration / num_iterations)

        for game_idx in range(games_per_iter):
            experiences = self_play_game(net, device, temperature=temperature)

            for exp_board, exp_probs, reward in experiences:
                buffer.add(exp_board, exp_probs, reward)

            if (game_idx + 1) % 10 == 0:
                print(f"  Games: {game_idx + 1}/{games_per_iter}, Buffer size: {len(buffer)}")

        # Training phase
        if len(buffer) >= batch_size:
            print("Training...")
            num_batches = min(100, len(buffer) // batch_size)
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0

            for _ in range(num_batches):
                batch = buffer.sample(batch_size)
                loss, p_loss, v_loss = train_batch(net, optimizer, batch, device)
                total_loss += loss
                total_policy_loss += p_loss
                total_value_loss += v_loss

            avg_loss = total_loss / num_batches
            avg_p_loss = total_policy_loss / num_batches
            avg_v_loss = total_value_loss / num_batches

            print(f"  Loss: {avg_loss:.4f} (Policy: {avg_p_loss:.4f}, Value: {avg_v_loss:.4f})")

        # Evaluation phase
        if (iteration + 1) % eval_interval == 0:
            # Save checkpoint for evaluation
            eval_path = "ur_eval_temp.pt"
            torch.save(net.state_dict(), eval_path)

            print("\nEvaluating against baseline policies...")
            elos, pairwise = evaluate_models(
                [eval_path], ["policy_random", "policy_aggressive"], num_games=50, max_workers=4
            )

            # Calculate win rate (ELO difference)
            neural_elo = elos.get(eval_path, 1000)
            baseline_avg_elo = elos.drop(eval_path).mean()
            elo_diff = neural_elo - baseline_avg_elo

            if elo_diff > best_win_rate:
                best_win_rate = elo_diff
                torch.save(net.state_dict(), "ur_best_model.pt")
                print(f"  New best model saved! (ELO diff: {best_win_rate:.0f})")

        # Save checkpoint
        if (iteration + 1) % save_interval == 0:
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_elo_diff": best_win_rate,
                },
                f"ur_checkpoint_{iteration + 1}.pt",
            )
            print("  Checkpoint saved!")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best ELO difference: {best_win_rate:.0f}")
    print("=" * 60)

    return net


if __name__ == "__main__":
    # Train the agent
    trained_net = train(num_iterations=500, games_per_iter=50, batch_size=64, eval_interval=10, save_interval=50)

    # Save final model
    torch.save(trained_net.state_dict(), "ur_final_model.pt")
    print("\nFinal model saved as 'ur_final_model.pt'")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    evaluate_models(
        ["ur_final_model.pt"],
        ["policy_random", "policy_aggressive", "policy_first", "policy_last"],
        num_games=200,
        max_workers=4,
    )
