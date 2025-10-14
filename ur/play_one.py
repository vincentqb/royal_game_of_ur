import curses
from time import sleep

import numpy as np
from game import (
    COMMON,
    N_BOARD,
    N_PLAYER,
    ROSETTE,
    create_board,
    determine_winner,
    execute_move,
    get_legal_moves,
    standardize_state,
    throw,
)
from policies import policy_human


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

    def show_pieces(self, board, *, current_piece=None, current_player=None):
        if self.screen is None:
            return
        for player in range(N_PLAYER):
            for i in [0, len(board[0]) - 1]:
                y, x = self.map(i, player)
                style = curses.A_BOLD
                label = str(int(board[player][i].item()))
                self.screen.addstr(y + 1, x + 2, label, style)

            pieces = np.nonzero(board[player, :-1])[-1].tolist()
            for i in range(len(pieces)):
                y, x = self.map(pieces[i], player)
                if player == current_player and pieces[i] == current_piece:
                    style = curses.A_REVERSE
                else:
                    style = curses.A_BOLD
                if 0 < i < N_BOARD - 1:
                    label = str(player)
                else:
                    label = str(int(board[player][i].item()))
                self.screen.addstr(y + 1, x + 2, label, style)
        self.screen.refresh()


def play(policies, board=None, screen=None):
    visual = VisualBoard(screen)

    player = 0  # Starting player
    board = create_board() if board is None else board
    winner = []
    iteration = 0

    states = {
        "winner": -1,
        "boards": [standardize_state(board, player)],
        "players": [player],
    }

    while True:
        visual.show_board()
        visual.show_pieces(board)

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
    # play([policy_random, policy_random])
    curses.wrapper(play_human)
