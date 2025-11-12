import random
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from game import (
    N_PLAYER,
    ROSETTE,
    create_board,
    determine_winner,
    execute_move,
    get_legal_moves,
    standardize_state,
    throw,
)
from loguru import logger
from play_many import play_many as evaluate_models
from policies import UrNet, select_move
from rich.box import HORIZONTALS
from rich.live import Live
from rich.progress import track
from rich.table import Table
from utils import configure_logger, dtype, parallel_map


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, maxlen=50_000):
        self.buffer = deque(maxlen=maxlen)

    def extend(self, experience):
        """Add experience to buffer."""
        self.buffer.extend(experience)

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
    player = random.randrange(N_PLAYER)
    winner = []
    experiences = []
    iteration = 0
    max_iterations = 1000

    net = net.to(device)

    while iteration < max_iterations:
        dice = throw()
        moves = get_legal_moves(board, player, dice)

        if moves:
            std_board = standardize_state(board, player)
            move, probs = select_move(
                net, std_board, player, moves, device=device, temperature=temperature, training=True
            )

            if move:
                experience = dict(
                    board=std_board.copy(),
                    player=player,
                    dice=dice,
                    start=move[0],
                    end=move[1],
                    probs=probs.cpu().numpy(),
                )
                experiences.append(experience)
                execute_move(board, player, *move)

                winner = determine_winner(board)
                if winner:
                    assert len(winner) == 1
                    for experience in experiences:
                        experience["reward"] = 1.0 if experience["player"] == winner[0] else -1.0
                    break

                if move[-1] not in ROSETTE:
                    # Not a rosette
                    player = (player + 1) % N_PLAYER
            else:
                player = (player + 1) % N_PLAYER
        else:
            player = (player + 1) % N_PLAYER

        iteration += 1
    return experiences


def train_batch(net, optimizer, batch, *, device):
    """Train on a batch of experiences."""
    if not batch:
        return 0.0, 0.0, 0.0, 0.0

    boards = np.stack([exp["board"].astype(np.float32) for exp in batch])
    boards = torch.from_numpy(boards.reshape(len(batch), -1)).to(device).to(dtype)
    rewards = torch.tensor([exp["reward"] for exp in batch], requires_grad=False, dtype=dtype, device=device)

    policy_logits, _ = net(boards)

    # Policy loss: KL divergence
    log_probs = torch.log_softmax(policy_logits, dim=1)
    action_indices = torch.tensor([exp["start"] for exp in batch], requires_grad=False, dtype=torch.long, device=device)

    selected_log_probs = log_probs[torch.arange(len(batch)), action_indices]
    policy_loss = -(selected_log_probs * rewards).mean()

    optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()

    return policy_loss.item()


class TableQueue:
    def __init__(self, *, maxlen=10):
        self.queue = deque(maxlen=maxlen)

    def __call__(self, row):
        self.queue.append(row)

        table = Table(box=HORIZONTALS)

        keys = []
        for row in self.queue:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        # keys = set(sum([list(row.keys()) for row in self.queue], []))

        for key in keys:
            table.add_column(key, justify="right")

        for row in self.queue:
            table.add_row(*[str(row[key]) if key in row else "" for key in keys])

        return table


def train(
    *,
    exp_dir,
    batch_size=25,
    num_iterations=500,
    num_epochs=1_000,
    save_interval=25,
):
    """
    Main training loop.

    Args:
        exp_dir: Path to experiment directory
        batch_size: Number of self-play games per iteration
        num_iterations: Number of training batches per iteration
        num_epochs: Number of training iterations
        save_interval: Save and evaluate model every N iterations

    Returns:
        net: Trained network
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    logger.trace(
        "Doing {num_epochs} epochs of {num_iterations} iterations of size {batch_size} on {device}",
        num_epochs=num_epochs,
        num_iterations=num_iterations,
        batch_size=batch_size,
        save_interval=save_interval,
        device=device,
    )

    net = UrNet(device=device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    buffer = ReplayBuffer()

    best_elo = 0.0
    best_model_path = exp_dir / "best_model.pt"

    table_epoch = TableQueue(maxlen=10)
    table_elo = TableQueue(maxlen=5)

    with Live() as live_elo, Live() as live_epoch:
        for iteration in track(range(num_epochs), description="Epoch"):
            net.eval()

            temperature = max(0.5, 3.0 - 2.5 * iteration / num_epochs)
            tasks = [(net, temperature, device) for _ in range(batch_size)]
            for experiences in parallel_map(self_play_game, tasks, description="Self-Play..."):
                logger.trace(
                    "length of experiences: {length}",
                    length=max(len(experience) for experience in experiences),
                    iteration=iteration,
                )
                buffer.extend(experiences)

            logger.trace("length of buffer: {length}", length=len(buffer), iteration=iteration)

            net.train()

            loss = 0.0
            for _ in range(num_iterations):
                batch = buffer.sample(batch_size)
                loss += train_batch(net, optimizer, batch, device=device)

            loss = loss / num_iterations
            logger.trace("Loss: {loss:.4f}", loss=loss, iteration=iteration)
            row_epoch = {"Iteration": str(iteration), "Loss": f"{loss:.0f}"}

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

                # Now evaluate saved model
                elos, pairwise = evaluate_models([checkpoint_path, "policy_random", "policy_aggressive"], show=False)
                neural_elo = elos[checkpoint_path.stem]

                # Update best model symlink if best
                if neural_elo > best_elo:
                    best_elo = neural_elo
                    if best_model_path.exists() or best_model_path.is_symlink():
                        best_model_path.unlink()
                    best_model_path.symlink_to(checkpoint_path.name)
                    logger.debug("New Best ELO: {elo:.0f}", elo=neural_elo)
                if best_elo > 0:
                    row_epoch["Best ELO"] = f"{best_elo:.0f}"
                live_elo.update(
                    table_elo({
                        "Iteration": str(iteration),
                        "ELO": f"{neural_elo:.0f}",
                        "Best": f"{best_elo:.0f}",
                    })
                )

            live_epoch.update(table_epoch(row_epoch))

    logger.debug("Best ELO Observed: {elo:.0f}", elo=best_elo)

    return checkpoint_path


if __name__ == "__main__":
    # Create experiment directory

    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/{experiment_id}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    configure_logger(exp_dir)
    logger.info("Experiment directory: {exp_dir}", exp_dir=exp_dir)

    # Train agent

    best_model_path = exp_dir / "best_model.pt"
    checkpoint_model_path = None
    try:
        checkpoint_model_path = train(exp_dir=exp_dir)
    except KeyboardInterrupt:
        logger.warning("Training interupted by user")

    # Final evaluation

    available = [
        "policy_random",
        "policy_aggressive",
        "policy_first",
        "policy_last",
    ]
    if best_model_path.exists():
        available.append(best_model_path)
    if checkpoint_model_path is not None and checkpoint_model_path.exists():
        available.append(checkpoint_model_path)

    evaluate_models(available, num_games=200)
