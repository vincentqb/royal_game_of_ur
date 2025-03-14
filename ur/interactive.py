import curses
from time import sleep

import numpy as np

from engine import (
    COMMON,
    N_BOARD,
    N_PLAYER,
    ROSETTE,
    create_board,
    determine_winner,
    execute_move,
    get_legal_moves,
    throw,
)

# Board size
ROWS = N_PLAYER + 1
COLS = len(COMMON)
# Common row
COMMON_ROW = 1


def convert_to_2d(positions, player=None):
    MAP = [i if i < COMMON_ROW else i + 1 for i in range(N_PLAYER)]
    out = []
    players = range(N_PLAYER) if player is None else range(player, player + 1)
    for position in positions:
        if position in COMMON:
            out.append((COMMON_ROW, position - min(COMMON)))
        elif position < min(COMMON):
            for p in players:
                out.append((MAP[p], min(COMMON) - 1 - position))
        elif position > max(COMMON):
            for p in players:
                out.append((MAP[p], min(COMMON) + 1 + N_BOARD - position))
    return out


def show_board(screen):
    # Get terminal size
    height, width = screen.getmaxyx()
    # Set board start position (centered)
    start_y = (height - ROWS * 2) // 2
    start_x = (width - COLS * 4) // 2
    # Draw the board
    for row in range(ROWS):
        for col in range(COLS):
            if row != COMMON_ROW and col in range(min(COMMON) - 1, N_BOARD - len(COMMON)):
                continue
            y, x = start_y + row * 2, start_x + col * 4
            screen.addstr(y, x, "+---+")
            screen.addstr(y + 1, x, "|   |")
            screen.addstr(y + 2, x, "+---+")

            # Mark Rosettes
            ROSETTE2D = set(convert_to_2d(ROSETTE))
            assert ROSETTE2D == {(0, 0), (2, 0), (1, 3), (2, 6), (0, 6)}, f"got {ROSETTE2D=}"
            if (row, col) in ROSETTE2D:
                screen.addstr(y + 1, x + 2, "★", curses.A_BOLD)


def show_info(screen, msg):
    # Get terminal size
    height, width = screen.getmaxyx()
    # Set board start position (centered)
    start_y = (height - ROWS * 2) // 2
    start_x = (width - COLS * 4) // 2
    screen.addstr(start_y - 1, start_x, msg)
    screen.clrtoeol()


def show_pieces(screen, board, current_piece, current_player):
    # Get terminal size
    height, width = screen.getmaxyx()
    # Set board start position (centered)
    start_y = (height - ROWS * 2) // 2
    start_x = (width - COLS * 4) // 2

    for player in range(N_PLAYER):
        pieces = np.nonzero(board[player, :-1])[-1].tolist()
        pieces2d = convert_to_2d(pieces, player=player)
        pieces2d = [(start_y + p[0] * 2 + 1, start_x + p[1] * 4 + 2) for p in pieces2d]
        for i in range(len(pieces)):
            style = curses.A_REVERSE if player == current_player and pieces[i] == current_piece else curses.A_BOLD
            label = str(player) if 0 < i < N_BOARD else str(int(board[player][i].item()))
            screen.addstr(pieces2d[i][0], pieces2d[i][1], label, style)


def play(screen):
    curses.curs_set(0)  # Hide cursor
    screen.erase()

    board = create_board()

    winner = []
    player = 0  # Starting player
    iteration = 0

    while True:
        screen.erase()

        show_board(screen)

        dice = throw()
        moves = get_legal_moves(board, player, dice)

        show_info(screen, f"Player {player} threw {dice}.")

        if moves:
            move_index = 0
            while True:
                show_pieces(screen, board, moves[move_index][0], player)
                screen.refresh()

                # Get user input
                key = screen.getch()
                if key == ord("q"):
                    break
                elif key == 9:  # Tab key to cycle through pieces with legal moves
                    move_index = (move_index + 1) % len(moves)
                elif key == curses.KEY_LEFT:
                    move_index = (move_index - 1) % len(moves)
                elif key == curses.KEY_RIGHT:
                    move_index = (move_index + 1) % len(moves)
                elif key == 10:  # Enter key to confirm move
                    move = moves[int(move_index)]
                    board = execute_move(board, player, *move)
                    break
            if key == ord("q"):
                show_info(screen, "Players quit.")
                break
            if move[-1] in ROSETTE:
                # Play again on rosettes
                continue

        player = (player + 1) % N_PLAYER
        winner = determine_winner(board)
        if winner:
            show_info(screen, f"Player {player} won.")
            break

        iteration += 1
        if iteration > 1000:
            show_info(screen, "Game is too long.")
            break
    screen.refresh()
    sleep(1)


if __name__ == "__main__":
    curses.wrapper(play)
