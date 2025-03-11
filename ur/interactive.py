import curses

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
ROWS, COLS = 3, 8


def convert_to_2d(positions, player=None):
    COMMON_POSITION = 1
    MAP = [i if i < COMMON_POSITION else i + 1 for i in range(N_PLAYER)]
    out = []
    for position in positions:
        if position in COMMON:
            out.append((COMMON_POSITION, position - min(COMMON)))
        elif position < min(COMMON):
            if player is None:
                for p in range(N_PLAYER):
                    out.append((MAP[p], min(COMMON) - 1 - position))
            else:
                out.append((MAP[player], min(COMMON) - 1 - position))
        elif position > max(COMMON):
            if player is None:
                for p in range(N_PLAYER):
                    out.append((MAP[p], min(COMMON) + 1 + N_BOARD - position))
            else:
                out.append((MAP[player], min(COMMON) + 1 + N_BOARD - position))
    return out


def draw_board(stdscr):
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()

    # Get terminal size
    height, width = stdscr.getmaxyx()
    # Set board start position (centered)
    start_y = (height - ROWS * 2) // 2
    start_x = (width - COLS * 4) // 2

    player = 0  # starting player
    board = create_board()

    winner = []
    iteration = 0

    while True:
        stdscr.clear()

        # Draw the board
        for row in range(ROWS):
            for col in range(COLS):
                y, x = start_y + row * 2, start_x + col * 4
                stdscr.addstr(y, x, "+---+")
                stdscr.addstr(y + 1, x, "|   |")
                stdscr.addstr(y + 2, x, "+---+")

                # Mark Rosettes
                ROSETTE2D = set(convert_to_2d(ROSETTE))
                assert ROSETTE2D == {(0, 0), (2, 0), (1, 3), (2, 6), (0, 6)}, f"got {ROSETTE2D=}"
                if (row, col) in ROSETTE2D:
                    stdscr.addstr(y + 1, x + 2, "★", curses.A_BOLD)

        dice = throw()
        stdscr.addstr(start_y - 1, start_x, f"Player {player} threw {dice}.")
        moves = get_legal_moves(board, player, dice)

        move_index = 0

        for p in range(N_PLAYER):
            tokens = np.nonzero(board[p, :-1])[-1].tolist()
            tokens = convert_to_2d(tokens, player=p)
            for i, (py, px) in enumerate(tokens):
                piece_y, piece_x = start_y + py * 2 + 1, start_x + px * 4 + 2
                style = curses.A_BOLD
                label = str(p) if 0 < i < N_BOARD else str(int(board[p][i].item()))
                stdscr.addstr(piece_y, piece_x, label, style)  # Highlight selected piece

        stdscr.addstr(start_y - 1, start_x, f"Player {player} threw {dice}.")
        stdscr.refresh()

        if moves:
            # Get user input
            while True:
                tokens = np.nonzero(board[player, :-1])[-1].tolist()
                tokens2d = convert_to_2d(tokens, player=player)
                for i in range(len(tokens)):
                    py, px = tokens2d[i]
                    piece_y, piece_x = start_y + py * 2 + 1, start_x + px * 4 + 2
                    style = curses.A_REVERSE if tokens[i] == moves[move_index][0] else curses.A_BOLD
                    label = str(player) if 0 < i < N_BOARD else str(int(board[player][i].item()))
                    stdscr.addstr(piece_y, piece_x, label, style)  # Highlight selected piece

                key = stdscr.getch()
                if key == ord("q"):
                    break
                if key == 9:  # Tab key to cycle through pieces with legal moves
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
                break
            if move[-1] in ROSETTE:
                # Play again on rosettes
                continue

        player = (player + 1) % N_PLAYER
        winner = determine_winner(board)
        if winner:
            stdscr.addstr(start_y - 1, start_x, f"Player {player} won.")
            break

        iteration += 1

        if iteration > 1000:
            stdscr.addstr(start_y - 1, start_x, "Game is too long.")
            break
    stdscr.refresh()


# Run the game loop
curses.wrapper(draw_board)
