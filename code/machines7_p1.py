import numpy as np
import random
import time
import copy
from itertools import product

# ===================== 상수 및 전역 변수 =====================
BOARD_ROWS = 4
BOARD_COLS = 4
EXPLORATION_WEIGHT = 1.5
WEIGHT_CHANGE = 0.04

CALC_TIME = [
    1,1,1,1,2,2,2,2,2,2,2,2,2,14,21,35,
    39,44,49,54,60,59,49,44,35,35,31,1,1,1,1,1
]
PIECES = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]

CENTER_CELLS = [(1, 1), (1, 2), (2, 1), (2, 2)]  # 0-indexed
CORNER_CELLS = [(0, 0), (0, 3), (3, 0), (3, 3)]

# ===================== 전략 보너스 함수 =====================

def center_bonus(location):
    return 0.2 if location in CENTER_CELLS else 0

def corner_bonus(location):
    return 0.1 if location in CORNER_CELLS else 0

def attribute_distribution_bonus(board, location, piece):
    if piece is None:
        return 0
    row, col = location
    bonus = 0
    lines = [
        [board[row, i] for i in range(4)],
        [board[i, col] for i in range(4)],
    ]
    if row == col:
        lines.append([board[i, i] for i in range(4)])
    if row + col == 3:
        lines.append([board[i, 3 - i] for i in range(4)])
    for line in lines:
        attrs = [PIECES[idx - 1] for idx in line if idx != 0]
        if not attrs:
            continue
        for i in range(4):
            cnt = sum(attr[i] == piece[i] for attr in attrs)
            if cnt >= 2:
                bonus -= 0.15
            elif cnt == 1:
                bonus += 0.05
    return bonus

def fork_bonus(board, location, piece):
    if piece is None:
        return 0
    temp_board = copy.deepcopy(board)
    temp_board[location[0], location[1]] = PIECES.index(piece) + 1
    win_count = 0
    # 가로
    if check_line([temp_board[location[0], i] for i in range(4)]):
        win_count += 1
    # 세로
    if check_line([temp_board[i, location[1]] for i in range(4)]):
        win_count += 1
    # 대각선
    if location[0] == location[1] and check_line([temp_board[i, i] for i in range(4)]):
        win_count += 1
    if location[0] + location[1] == 3 and check_line([temp_board[i, 3 - i] for i in range(4)]):
        win_count += 1
    # 2x2 subgrid
    for dr in [0, -1]:
        for dc in [0, -1]:
            r, c = location[0] + dr, location[1] + dc
            if 0 <= r <= 2 and 0 <= c <= 2:
                subgrid = [temp_board[r, c], temp_board[r, c+1], temp_board[r+1, c], temp_board[r+1, c+1]]
                if 0 not in subgrid:
                    characteristics = [PIECES[idx - 1] for idx in subgrid]
                    for i in range(4):
                        if len(set(char[i] for char in characteristics)) == 1:
                            win_count += 1
    return 0.3 * (win_count - 1) if win_count >= 2 else 0

def immediate_win(board, location, piece):
    temp_board = copy.deepcopy(board)
    temp_board[location[0], location[1]] = PIECES.index(piece) + 1
    return check_win(temp_board)

def immediate_lose(board, able_pieces):
    # 상대가 두면 바로 이기는 곳이 있는지 체크
    for piece in able_pieces:
        for row in range(4):
            for col in range(4):
                if board[row, col] == 0:
                    temp_board = copy.deepcopy(board)
                    temp_board[row, col] = PIECES.index(piece) + 1
                    if check_win(temp_board):
                        return (row, col, piece)
    return None

def strategic_bonus(board, location, piece):
    bonus = 0
    bonus += center_bonus(location)
    bonus += corner_bonus(location)
    bonus += attribute_distribution_bonus(board, location, piece)
    bonus += fork_bonus(board, location, piece)
    return bonus

# ===================== 유틸리티 함수 =====================

def mbti_str(piece):
    return ''.join([
        'I' if piece[0] == 0 else 'E',
        'N' if piece[1] == 0 else 'S',
        'T' if piece[2] == 0 else 'F',
        'P' if piece[3] == 0 else 'J'
    ])

def print_board(board):
    for row in board:
        print([
            mbti_str(PIECES[col-1]) if col != 0 else '    '
            for col in row
        ])

def print_child_data(node):
    for child in node.childNodes:
        print("..........................", child.depth)
        print_board(child.board)
        print('x:', child.x)
        print('n:', child.n)
        print('done:', child.done)
        print(child.data)

def new_board():
    return np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)

def new_pieces():
    return [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]

def detect_child(childNodes, data):
    for child in childNodes:
        if child.data == data:
            return child
    return None

def detect_placed_location(origin_board, new_board):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if origin_board[row][col] != new_board[row][col]:
                return (row, col)
    return None

def detect_able_locs(board):
    return [(row, col) for row, col in product(range(4), range(4)) if board[row][col] == 0]

def check_line(line):
    if 0 in line:
        return False
    characteristics = np.array([PIECES[piece_idx - 1] for piece_idx in line])
    for i in range(4):
        if len(set(characteristics[:, i])) == 1:
            return True
    return False

def check_2x2_subgrid_win(board):
    for r in range(BOARD_ROWS - 1):
        for c in range(BOARD_COLS - 1):
            subgrid = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
            if 0 not in subgrid:
                characteristics = [PIECES[idx - 1] for idx in subgrid]
                for i in range(4):
                    if len(set(char[i] for char in characteristics)) == 1:
                        return True
    return False

def check_win(board):
    for col in range(BOARD_COLS):
        if check_line([board[row][col] for row in range(BOARD_ROWS)]):
            return True
    for row in range(BOARD_ROWS):
        if check_line([board[row][col] for col in range(BOARD_COLS)]):
            return True
    if check_line([board[i][i] for i in range(BOARD_ROWS)]) or \
       check_line([board[i][BOARD_ROWS - i - 1] for i in range(BOARD_ROWS)]):
        return True
    if check_2x2_subgrid_win(board):
        return True
    return False

# ===================== 트리 노드 클래스 =====================

class SelectNode:
    def __init__(self, parent, piece, able_pieces, mine, depth=0):
        self.parent = parent
        self.data = piece
        self.able_pieces = able_pieces
        self.mine = mine
        self.depth = depth
        self.board = parent.board
        self.childNodes = []
        self.x = 0.0
        self.n = 0
        self.done = 0

    def expand(self):
        for row, col in detect_able_locs(self.parent.board):
            if not detect_child(self.childNodes, (row, col)):
                new_board = copy.deepcopy(self.parent.board)
                new_board[row][col] = PIECES.index(self.data) + 1
                new_able_pieces = copy.deepcopy(self.able_pieces)
                new_node = PlaceNode(self, new_board, (row, col), new_able_pieces, not self.mine, self.depth + 1)
                self.childNodes.append(new_node)

class PlaceNode:
    def __init__(self, parent, board, location, able_pieces, mine, depth=0):
        self.parent = parent
        self.board = board
        self.data = location
        self.able_pieces = able_pieces
        self.mine = mine
        self.depth = depth
        self.childNodes = []
        self.x = 0.0
        self.n = 0
        self.done = 0

    def expand(self):
        for piece in self.able_pieces:
            if not detect_child(self.childNodes, piece):
                new_able_pieces = copy.deepcopy(self.able_pieces)
                new_able_pieces.remove(piece)
                new_node = SelectNode(self, piece, new_able_pieces, self.mine, self.depth + 1)
                self.childNodes.append(new_node)

# ===================== 플레이어 클래스 =====================

class P1():
    head = PlaceNode(None, new_board(), None, new_pieces(), False)
    current_location = None
    current_board = new_board()
    current_piece = None

    def __init__(self, board, available_pieces):
        self.board = board
        self.available_pieces = available_pieces
        if len(available_pieces) == 16:
            P1.reset()

    def select_piece(self):
        head_set([P1.current_location])
        selected_piece = minimax_select_piece(self.board, self.available_pieces)
        print("P2: selected", selected_piece, mbti_str(selected_piece))
        P1.current_piece = selected_piece
        P1.current_board = copy.deepcopy(self.board)
        return selected_piece

    def place_piece(self, selected_piece):
        placed_loc = detect_placed_location(P1.current_board, self.board)
        head_set([P1.current_piece, placed_loc, selected_piece])
        position = monte_carlo(P1.head)
        print("P1: placed at", position)
        P1.current_location = position
        return position

    @staticmethod
    def reset():
        P1.head = PlaceNode(None, new_board(), None, new_pieces(), False)

# ===================== MCTS 핵심 함수 =====================

def head_set(data_list):
    if P1.head.parent is None:
        data_list = [data_list[-1]]
    for data in data_list:
        P1.head.expand()
        P1.head = detect_child(P1.head.childNodes, data)
        if not P1.head:
            print("head_set 실패")

def monte_carlo(root):
    begin = time.time()
    if not root.done:
        while time.time() < begin + CALC_TIME[root.depth]:
            if root.done:
                print('P1: all possible cases calculated.')
                if root.done == 1:
                    print('P1: sure win!')
                else:
                    print('P1: might lose.')
                break
            node = root
            while node.childNodes:
                node = select_to_roll_child(node)
            x = simulate_game(node)
            backpropagate(node, x)
    best_child = select_best_child(root)
    return best_child.data

def select_to_roll_child(node):
    def ucb1_score(child):
        if child.n == 0:
            return float('inf')
        dont_care = 999 if child.done == 1 else 0
        exploitation = abs(child.x)
        exploration = (EXPLORATION_WEIGHT - WEIGHT_CHANGE * child.depth) * np.sqrt(np.log(node.n) / child.n)
        # 전략 보너스
        bonus = 0
        if isinstance(child, PlaceNode) and child.data is not None:
            # SelectNode에서 PlaceNode로 넘어갈 때, piece정보는 parent.data
            parent = child.parent
            piece = parent.data if parent else None
            bonus += strategic_bonus(child.board, child.data, piece)
            # 즉시승
            if piece and immediate_win(child.board, child.data, piece):
                return 1000
            # 즉시패(막아야 할 곳)
            if piece:
                lose_info = immediate_lose(child.board, child.able_pieces)
                if lose_info and lose_info[0:2] == child.data:
                    bonus += 500
        return exploitation + exploration + bonus - dont_care
    return max(node.childNodes, key=ucb1_score)

def select_best_child(node):
    def node_score(child):
        score = child.x + child.done
        if isinstance(child, PlaceNode) and child.data is not None:
            parent = child.parent
            piece = parent.data if parent else None
            score += strategic_bonus(child.board, child.data, piece)
            if piece and immediate_win(child.board, child.data, piece):
                return 1000
            if piece:
                lose_info = immediate_lose(child.board, child.able_pieces)
                if lose_info and lose_info[0:2] == child.data:
                    score += 500
        return score
    return max(node.childNodes, key=node_score)

def simulate_game(node):
    node.expand()
    temp_board = copy.deepcopy(node.board)
    temp_able_pieces = copy.deepcopy(node.able_pieces)
    temp_able_locs = detect_able_locs(temp_board)
    current_mine = node.mine

    if check_win(temp_board):
        node.done = 1 if current_mine else -1
    if not temp_able_pieces:
        node.done = 0.0001

    if isinstance(node, SelectNode):
        current_mine = not current_mine
        row, col = random.choice(temp_able_locs)
        temp_able_locs.remove((row, col))
        temp_board[row][col] = PIECES.index(node.data) + 1

    while not check_win(temp_board):
        if not temp_able_pieces:
            return 0
        current_mine = not current_mine
        piece = random.choice(temp_able_pieces)
        row, col = random.choice(temp_able_locs)
        temp_able_pieces.remove(piece)
        temp_able_locs.remove((row, col))
        temp_board[row][col] = PIECES.index(piece) + 1

    return 1 if current_mine == node.mine else -1

def backpropagate(node, x):
    temp_x = x
    while node is not None:
        if not node.mine:
            temp_x = -x
        adder = 1 if isinstance(node, PlaceNode) == node.mine else -1
        if node.childNodes:
            trigger = True
            for child in node.childNodes:
                if node.done:
                    trigger = False
                    break
                if child.done == adder:
                    node.done = adder
                    trigger = False
                    break
                if child.done != -adder:
                    trigger = False
                    continue
            if trigger:
                node.done = -adder
        node.x = (node.n * node.x + temp_x) / (node.n + 1)
        node.n += 1
        node = node.parent

# =================== Minimax 핵심 함수 ===================
def minimax_select_piece(board, available_pieces):
    able_locs = detect_able_locs(board)
    safe_pieces = []
    for piece in available_pieces:
        is_safe = True
        for loc in able_locs:
            temp_board = copy.deepcopy(board)
            temp_board[loc[0], loc[1]] = PIECES.index(piece) + 1
            if check_win(temp_board):  # 상대가 이길 수 있는 조각이면
                is_safe = False
                break
        if is_safe:
            safe_pieces.append(piece)
    if safe_pieces:
        # 안전한 조각 중에서 선택 (여러 전략을 쓸 수 있음)
        return random.choice(safe_pieces)
    # 만약 모든 조각이 위험하면, 기존 방식대로 선택
    return random.choice(available_pieces)
