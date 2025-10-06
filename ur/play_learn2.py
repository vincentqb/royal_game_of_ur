import random
from collections import deque

import numpy as np
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
    standardize_state,
    throw,
)


class UrNet(nn.Module):
    """Neural network for policy and value prediction."""

    def __init__(self, input_size=(N_PLAYER * (N_BOARD + 2)), hidden_size=128):
        super().__init__()

        # Input is flattened board state
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )

        # Policy head: outputs logits for each board position (potential moves)
        self.policy = nn.Linear(hidden_size // 2, N_BOARD + 2)

        # Value head: outputs win probability for current player
        self.value = nn.Sequential(nn.Linear(hidden_size // 2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, x):
        features = self.shared(x)
        policy_logits = self.policy(features)
        value = self.value(features)
        return policy_logits, value


class AlphaUrAgent:
    """AlphaGo-style agent for Royal Game of Ur."""

    def __init__(self, lr=0.001, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.net = UrNet().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def board_to_tensor(self, board):
        """Convert board state to tensor."""
        # Normalize piece counts
        board_norm = board.astype(np.float32) / N_PIECE
        return torch.from_numpy(board_norm.flatten()).to(self.device)

    def get_move_mask(self, board, player, dice, moves):
        """Create mask for legal moves based on dice roll.

        Returns:
            mask: Tensor with 0 for legal starting positions, -inf for illegal
            move_map: Dict mapping board positions to actual moves
        """
        mask = torch.full((N_BOARD + 2,), float("-inf"), device=self.device)
        move_map = {}

        for start, end in moves:
            mask[start] = 0.0  # Legal starting position
            move_map[start] = (start, end)

        return mask, move_map

    def select_move(self, board, player, dice, moves, temperature=1.0, training=True):
        """Select move using policy network with dice-based masking.

        Args:
            board: Current board state
            player: Current player
            dice: Dice roll value
            moves: List of legal (start, end) moves
            temperature: Exploration parameter
            training: If True, sample; if False, take argmax

        Returns:
            move: Selected (start, end) move
            value: Estimated value of position
            probs: Action probabilities (for training)
        """
        if not moves:
            return None, 0.0, None

        # Get board tensor
        board_tensor = self.board_to_tensor(board).unsqueeze(0)

        with torch.no_grad():
            policy_logits, value = self.net(board_tensor)
            policy_logits = policy_logits.squeeze(0)

        # Apply mask based on legal moves
        mask, move_map = self.get_move_mask(board, player, dice, moves)
        masked_logits = policy_logits + mask

        # Apply temperature and get probabilities
        probs = torch.softmax(masked_logits / temperature, dim=0)

        if training:
            # Sample from distribution
            action_idx = torch.multinomial(probs, 1).item()
        else:
            # Take best action
            action_idx = torch.argmax(probs).item()

        move = move_map.get(action_idx)
        if move is None:
            # Fallback to random legal move if mapping fails
            move = random.choice(moves)

        return move, value.item(), probs.cpu().numpy()


def policy_neural(board, player, moves, agent, temperature=1.0, **kwargs):
    """Policy function compatible with play_one interface."""
    move, _, _ = agent.select_move(board, player, 0, moves, temperature, training=True)
    return move


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)

    def add(self, board, player, move_probs, reward):
        """Add experience to buffer."""
        self.buffer.append({"board": board.copy(), "player": player, "move_probs": move_probs, "reward": reward})

    def sample(self, batch_size):
        """Sample batch from buffer."""
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def self_play_game(agent, temperature=1.0):
    """Play one game against itself.

    Returns:
        experiences: List of (board, player, move_probs, reward) tuples
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
            # Get standardized state from current player's perspective
            std_board = standardize_state(board, player)
            move, value, probs = agent.select_move(std_board, player, dice, moves, temperature, training=True)

            if move:
                experiences.append((std_board, player, probs))
                execute_move(board, player, *move)

                # Check if move was on rosette (play again)
                if move[-1] not in [4, 8, 14]:  # Not a rosette
                    player = (player + 1) % N_PLAYER
            else:
                player = (player + 1) % N_PLAYER
        else:
            player = (player + 1) % N_PLAYER

        # Check for winner
        winner = determine_winner(board)
        if winner:
            # Assign rewards: +1 for winner, -1 for loser
            rewards = []
            for _, exp_player, _ in experiences:
                if exp_player == winner[0]:
                    rewards.append(1.0)
                else:
                    rewards.append(-1.0)

            # Create training examples
            training_data = []
            for (exp_board, exp_player, exp_probs), reward in zip(experiences, rewards):
                training_data.append((exp_board, exp_player, exp_probs, reward))

            return training_data

        iteration += 1

    # Game too long, no winner
    return []


def train_batch(agent, batch):
    """Train on a batch of experiences."""
    if not batch:
        return 0.0, 0.0, 0.0

    boards = []
    move_probs_list = []
    rewards = []

    for exp in batch:
        boards.append(agent.board_to_tensor(exp["board"]))
        move_probs_list.append(torch.from_numpy(exp["move_probs"]).to(agent.device))
        rewards.append(exp["reward"])

    boards = torch.stack(boards)
    target_probs = torch.stack(move_probs_list)
    rewards = torch.from_numpy(rewards).to(agent.device)

    # Forward pass
    policy_logits, values = agent.net(boards)
    values = values.squeeze()

    # Policy loss: KL divergence between old and new policy
    log_probs = torch.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_probs * log_probs).sum(dim=1).mean()

    # Value loss: MSE with actual game outcome
    value_loss = ((values - rewards) ** 2).mean()

    # Total loss
    loss = policy_loss + value_loss

    # Backprop
    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.net.parameters(), 1.0)
    agent.optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item()


def evaluate_agent(agent, opponent_policy, num_games=50):
    """Evaluate agent against a baseline policy.

    Returns:
        win_rate: Percentage of games won
    """
    from play_one import play, policy_random

    wins = 0

    # Create policy function for agent
    def agent_policy(**kwargs):
        return policy_neural(agent=agent, temperature=0.1, **kwargs)

    for _ in range(num_games):
        # Play as player 0
        result = play([agent_policy, opponent_policy], board=None, screen=None)
        if result["winner"] == 0:
            wins += 1

        # Play as player 1
        result = play([opponent_policy, agent_policy], board=None, screen=None)
        if result["winner"] == 1:
            wins += 1

    return wins / (num_games * 2)


def train(num_iterations=500, games_per_iter=50, batch_size=64, eval_interval=10, save_interval=50):
    """Main training loop.

    Args:
        num_iterations: Number of training iterations
        games_per_iter: Self-play games per iteration
        batch_size: Training batch size
        eval_interval: Evaluate every N iterations
        save_interval: Save model every N iterations
    """
    from play_one import policy_aggressive, policy_random

    print("Initializing AlphaUr training...")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    agent = AlphaUrAgent(lr=0.001)
    buffer = ReplayBuffer(max_size=50000)

    best_win_rate = 0.0

    for iteration in range(num_iterations):
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'=' * 60}")

        # Self-play phase
        print(f"Self-play: generating {games_per_iter} games...")
        temperature = max(0.5, 1.0 - iteration / num_iterations)  # Decay temperature

        for game_idx in range(games_per_iter):
            experiences = self_play_game(agent, temperature=temperature)

            for exp_board, exp_player, exp_probs, reward in experiences:
                buffer.add(exp_board, exp_player, exp_probs, reward)

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
                loss, p_loss, v_loss = train_batch(agent, batch)
                total_loss += loss
                total_policy_loss += p_loss
                total_value_loss += v_loss

            avg_loss = total_loss / num_batches
            avg_p_loss = total_policy_loss / num_batches
            avg_v_loss = total_value_loss / num_batches

            print(f"  Loss: {avg_loss:.4f} (Policy: {avg_p_loss:.4f}, Value: {avg_v_loss:.4f})")

        # Evaluation phase
        if (iteration + 1) % eval_interval == 0:
            print("\nEvaluating against random policy...")
            win_rate_random = evaluate_agent(agent, policy_random, num_games=25)
            print(f"  Win rate vs random: {win_rate_random:.2%}")

            print("Evaluating against aggressive policy...")
            win_rate_aggressive = evaluate_agent(agent, policy_aggressive, num_games=25)
            print(f"  Win rate vs aggressive: {win_rate_aggressive:.2%}")

            avg_win_rate = (win_rate_random + win_rate_aggressive) / 2

            if avg_win_rate > best_win_rate:
                best_win_rate = avg_win_rate
                torch.save(agent.net.state_dict(), "ur_best_model.pt")
                print(f"  New best model saved! (Win rate: {best_win_rate:.2%})")

        # Save checkpoint
        if (iteration + 1) % save_interval == 0:
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": agent.net.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "best_win_rate": best_win_rate,
                },
                f"ur_checkpoint_{iteration + 1}.pt",
            )
            print("  Checkpoint saved!")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best win rate achieved: {best_win_rate:.2%}")
    print("=" * 60)

    return agent


def load_agent(checkpoint_path):
    """Load a trained agent from checkpoint."""
    agent = AlphaUrAgent()
    checkpoint = torch.load(checkpoint_path, map_location=agent.device)

    if "model_state_dict" in checkpoint:
        agent.net.load_state_dict(checkpoint["model_state_dict"])
    else:
        agent.net.load_state_dict(checkpoint)

    agent.net.eval()
    return agent


if __name__ == "__main__":
    # Train the agent
    trained_agent = train(num_iterations=500, games_per_iter=50, batch_size=64, eval_interval=10, save_interval=50)

    # Save final model
    torch.save(trained_agent.net.state_dict(), "ur_final_model.pt")
    print("\nFinal model saved as 'ur_final_model.pt'")
