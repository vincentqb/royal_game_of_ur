import curses
import random
from time import sleep

import numpy as np

N_PLAYER = 2
N_PIECE = 7
N_BOARD = 14
SAFE = sorted([8])
ROSETTE = sorted([4, 8, 14])
COMMON = sorted(range(5, 13))


def create_board():
    # pieces start on [0] and end on [-1]
    board = np.zeros((N_PLAYER, N_BOARD + 2), dtype=np.uint8)
    board[:, 0] = N_PIECE
    return board


def throw():
    return np.random.randint(0, 2, size=4, dtype=np.uint8).sum().item()


def get_legal_moves(board, player, dice):
    if dice == 0:
        return []

    enemies = [p for p in range(N_PLAYER) if p != player]
    inds = np.nonzero(board[player, :-1])[-1].tolist()

    moves = []
    for start in inds:
        end = start + dice
        if start > max(COMMON) and end > N_BOARD + 1:
            # Player needs exact throw for last two squares
            continue
        end = min(end, N_BOARD + 1)
        if end < N_BOARD + 1 and board[[player], end].sum() > 0:
            # Player has a piece there
            continue
        if end in SAFE and board[enemies, end].sum() > 0:
            # Ennemy on safe rosette
            continue
        moves.append((start, end))

    return moves


def determine_winner(board):
    # Empty if no winners
    return np.nonzero(board[:, -1] == N_PIECE)[0].tolist()


def execute_move(board, player, start, end):
    enemies = [p for p in range(N_PLAYER) if p != player]
    if end in COMMON and board[enemies, end].sum() > 0:
        enemy = np.nonzero(board[:, end])[0].item()
        board[enemy, end] -= 1
        board[enemy, 0] += 1
    board[player, start] -= 1
    board[player, end] += 1


class VisualBoard:
    def __init__(self, screen):
        self.screen = screen
        if self.screen is None:
            return
        curses.curs_set(0)  # Hide cursor
        self.screen.erase()

        self.logs = []

        # Get terminal size
        height, width = screen.getmaxyx()
        # Board size
        self.ROWS = N_PLAYER + 1
        self.COLS = len(COMMON)
        # Set board start position (centered)
        self.start_y = (height - self.ROWS * 2) // 2
        self.start_x = (width - self.COLS * 4) // 2
        # Common row
        self.COMMON_ROW = 1

    def map(self, position, player):
        MAP = [i if i < self.COMMON_ROW else i + 1 for i in range(N_PLAYER)]
        if position in COMMON:
            y = self.COMMON_ROW
            x = position - min(COMMON)
        elif position < min(COMMON):
            y = MAP[player]
            x = min(COMMON) - 1 - position
        elif position > max(COMMON):
            y = MAP[player]
            x = min(COMMON) + 1 + N_BOARD - position
        y = self.start_y + y * 2
        x = self.start_x + x * 4
        return y, x

    def show_grid(self):
        if self.screen is None:
            return
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if row != self.COMMON_ROW and col in range(min(COMMON) - 1, N_BOARD - len(COMMON)):
                    continue
                y, x = self.start_y + row * 2, self.start_x + col * 4
                self.screen.addstr(y, x, "+---+")
                self.screen.addstr(y + 1, x, "|   |")
                self.screen.addstr(y + 2, x, "+---+")
        self.end_y = y + 2

    def show_rosette(self):
        for rosette in ROSETTE:
            for player in range(N_PLAYER):
                y, x = self.map(rosette, player)
                self.screen.addstr(y + 1, x + 2, "★", curses.A_BOLD)

    def show_board(self):
        if self.screen is None:
            return
        self.screen.erase()
        self.show_grid()
        self.show_rosette()

    def show_info(self, msg):
        if self.screen is None:
            return
        self.logs.append(msg)
        self.logs = self.logs[-1:]
        for i, msg in enumerate(self.logs[::-1]):
            self.screen.addstr(self.end_y + 1 + i, self.start_x, msg)
        # self.screen.addstr(self.start_y - 1, self.start_x, msg)
        self.screen.clrtoeol()

    def show_pieces(self, board, current_piece, current_player):
        if self.screen is None:
            return
        for player in range(N_PLAYER):
            pieces = np.nonzero(board[player, :])[-1].tolist()
            for i in range(len(pieces)):
                y, x = self.map(pieces[i], player)
                style = curses.A_REVERSE if player == current_player and pieces[i] == current_piece else curses.A_BOLD
                label = str(player) if 0 < i < N_BOARD else str(int(board[player][i].item()))
                self.screen.addstr(y + 1, x + 2, label, style)
        self.screen.refresh()


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


def standardize_state(board, player):
    # return np.concat([np.array([player], dtype=np.uint8), board.flatten()], dtype=np.uint8)
    rows = list(range(N_PLAYER))
    rows = rows[-player:] + rows[:player]  # rotate
    return board[rows, :]


def play(policies, screen=None):
    visual = VisualBoard(screen)

    player = 0  # Starting player
    board = create_board()
    winner = []
    iteration = 0

    states = {
        "winner": -1,
        "boards": [standardize_state(board, player)],
        "players": [player],
    }

    while True:
        visual.show_board()

        dice = throw()
        moves = get_legal_moves(board, player, dice)

        visual.show_info(f"Player {player} threw {dice}.")

        if moves:
            move = policies[player](board=board, player=player, moves=moves, visual=visual)
            if move == -1:
                visual.show_info("Players quit.")
                break
            execute_move(board, player, *move)
            states["boards"].append(standardize_state(board, player))
            states["players"].append(player)
            if move[-1] in ROSETTE:
                # Play again on rosettes
                continue

        player = (player + 1) % N_PLAYER
        winner = determine_winner(board)
        if winner:
            visual.show_info(f"Player {player} won.")
            states["winner"] = winner[0]
            break

        iteration += 1
        if iteration > 1000:
            visual.show_info("Game is too long.")
            break

    states["boards"] = np.stack(states["boards"], dtype=np.uint8)
    states["players"] = np.array(states["players"], dtype=np.uint8)
    return states


def play_human(screen):
    play([policy_human, policy_human], screen=screen)
    screen.refresh()
    sleep(1)


if __name__ == "__main__":
    curses.wrapper(play_human)
