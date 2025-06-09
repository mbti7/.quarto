import numpy as np
import random
from collections import defaultdict
from itertools import product
import math
import copy

# --- 파라미터 ---
MCTS_ITERATIONS = 1000  # MCTS 반복 횟수
SWITCH_POINT = 8        # Minimax로 전환하는 남은 말 개수 기준

# --- 보드 상수 ---
BOARD_ROWS = 4
BOARD_COLS = 4
pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # 16개의 모든 말

PLAYER = 2
isFirst = True

# --- Minimax용 DP 테이블 ---
DP_table = {}

# --- 위치 보너스 상수 ---
CENTER_POS = {(1, 1), (1, 2), (2, 1), (2, 2)}  # 중앙 위치
CORNER_POS = {(0, 0), (0, 3), (3, 0), (3, 3)}  # 코너 위치

# ===============================
# 유틸리티 함수
# ===============================

def restart_game():
    global DP_table
    DP_table = {}

def get(board, x, y):
    piece_idx = board[x][y] - 1
    if piece_idx == -1:
        return None
    else:
        return pieces[piece_idx]

def get_piece_idx(piece):
    return pieces.index(piece) + 1

def is_board_full(available_places):
    return not available_places

def get_position_bonus(row, col):
    bonus = 0.0
    if (row, col) in CENTER_POS:
        bonus += 0.2
    elif (row, col) in CORNER_POS:
        bonus += 0.1
    return bonus

def check_line(pieces):
    # 한 줄이 승리 조건을 만족하는지 확인
    return None not in pieces and any(
        all(piece[i] == pieces[0][i] for piece in pieces) for i in range(4)
    )

def get_fork_bonus(board, piece, row, col):
    # 해당 위치에 piece를 뒀을 때 동시에 몇 개의 승리 조건을 만족하는지 카운트
    win_count = 0
    board[row][col] = get_piece_idx(piece)
    # 가로
    if check_line([get(board, row, j) for j in range(BOARD_COLS)]): win_count += 1
    # 세로
    if check_line([get(board, i, col) for i in range(BOARD_ROWS)]): win_count += 1
    # 주대각선
    if row == col and check_line([get(board, i, i) for i in range(BOARD_ROWS)]): win_count += 1
    # 부대각선
    if row + col == BOARD_ROWS - 1 and check_line([get(board, i, BOARD_ROWS - 1 - i) for i in range(BOARD_ROWS)]): win_count += 1
    # 2x2 영역
    for i in range(max(0, row - 1), min(BOARD_ROWS - 2, row) + 1):
        for j in range(max(0, col - 1), min(BOARD_COLS - 2, col) + 1):
            if check_line([
                get(board, i, j), get(board, i, j + 1),
                get(board, i + 1, j), get(board, i + 1, j + 1)
            ]):
                win_count += 1
    board[row][col] = 0
    # 2개 이상이면 포크, 1개면 그냥 승리
    return 0.1 * max(0, win_count - 1)  # 2개면 0.1, 3개면 0.2, ...

# ===============================
# 승리 체크 함수
# ===============================

def check_win_with_piece(board, piece, x, y):
    # piece를 임시로 놓고 승리 여부 확인
    if piece not in pieces:
        raise ValueError(f"Piece {piece} is not available")
    board[x][y] = get_piece_idx(piece)
    flag = check_win(board, x, y)
    board[x][y] = 0
    return flag

def check_win(board, x, y):
    def check_equal_attributes(pieces):
        return any(all(piece[i] == pieces[0][i] for piece in pieces) for i in range(4))

    def check_line_local(pieces):
        return None not in pieces and check_equal_attributes(pieces)

    def get_2x2_pieces(i, j):
        return [
            get(board, i, j), get(board, i, j + 1),
            get(board, i + 1, j), get(board, i + 1, j + 1)
        ]

    # 가로줄
    row_pieces = [get(board, x, j) for j in range(BOARD_COLS)]
    if check_line_local(row_pieces):
        return True

    # 세로줄
    col_pieces = [get(board, i, y) for i in range(BOARD_ROWS)]
    if check_line_local(col_pieces):
        return True

    # 주대각선 (좌상단→우하단)
    if x == y:
        diag_pieces = [get(board, i, i) for i in range(BOARD_ROWS)]
        if check_line_local(diag_pieces):
            return True

    # 부대각선 (우상단→좌하단)
    if x + y == BOARD_ROWS - 1:
        anti_diag_pieces = [get(board, i, BOARD_ROWS - 1 - i) for i in range(BOARD_ROWS)]
        if check_line_local(anti_diag_pieces):
            return True

    # 2x2 영역
    for i in range(max(0, x - 1), min(BOARD_ROWS - 2, x) + 1):
        for j in range(max(0, y - 1), min(BOARD_COLS - 2, y) + 1):
            if check_line_local(get_2x2_pieces(i, j)):
                return True

    return False

# ===============================
# 플레이어 클래스
# ===============================

class P2():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board  # 0: 빈칸 / 1~16: 말
        self.available_pieces = available_pieces
        self.available_places = self.get_available_places()
    
    def select_piece(self):
        global isFirst

        if isFirst:
            isFirst = False
            return random.choice(self.pieces)

        if len(self.available_pieces) > SWITCH_POINT:
            return self._select_piece_mcts()
        else:
            return self._select_piece_minimax()

    def _select_piece_mcts(self):
        tree = MCTS()
        board = Board(self.board, PLAYER, None, self.available_places, self.available_pieces)
        node = Node(board)
        
        # 상대가 이길 수 있는 piece 제외
        tree.children[node] = []
        for piece in self.available_pieces:
            if not self._is_opponent_winning_piece(board, piece):
                next_board = copy.deepcopy(board)
                next_board.select(piece)
                next_node = Node(next_board)
                tree.children[node].append(next_node)

        if not tree.children[node]:
            return random.choice(self.available_pieces)
        if len(tree.children[node]) == 1:
            return tree.children[node][0].board_state.selected_piece

        # MCTS 실행
        reward = tree._simulate(node)
        tree._backpropagate([node], reward)

        for _ in range(MCTS_ITERATIONS):
            tree.do_rollout(node)

        best_node = tree.choose(node)
        return best_node.board_state.selected_piece

    def _select_piece_minimax(self):
        best_piece = None
        best_value = -float('inf')

        for piece in self.available_pieces:
            eval = self.minmax_alpha_beta(
                self.board, 
                [p for p in self.available_pieces if p != piece],
                -float('inf'), float('inf'),
                False, piece, None
            )
            if eval > best_value:
                best_value = eval
                best_piece = piece
                if best_value == 10:
                    break

        return best_piece

    def _is_opponent_winning_piece(self, board, piece):
        for row, col in self.available_places:
            if check_win_with_piece(board, piece, row, col):
                return True
        return False

    def place_piece(self, selected_piece):
        if len(self.available_pieces) > SWITCH_POINT:
            return self._place_piece_mcts(selected_piece)
        else:
            return self._place_piece_minimax(selected_piece)

    def _place_piece_mcts(self, selected_piece):
        tree = MCTS()
        board = Board(self.board, PLAYER, selected_piece, self.available_places, self.available_pieces)
        node = Node(board)

        # 즉시 승리 가능한 위치 확인
        for row, col in self.available_places:
            if check_win_with_piece(board, selected_piece, row, col):
                return row, col
        
        # MCTS 실행
        for _ in range(MCTS_ITERATIONS):
            tree.do_rollout(node)

        best_node = tree.choose(node)
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if best_node.board_state[row][col] == get_piece_idx(selected_piece):
                    return row, col

    def _place_piece_minimax(self, selected_piece):
        best_move = None
        best_value = -float('inf')

        for row, col in product(range(4), range(4)):
            if self.board[row][col] == 0:
                self.board[row][col] = get_piece_idx(selected_piece)
                eval = self.minmax_alpha_beta(
                    self.board,
                    self.available_pieces,
                    -float('inf'), float('inf'),
                    True, None, (row, col)
                )
                # 보너스 추가
                bonus = get_position_bonus(row, col) + get_fork_bonus(self.board, selected_piece, row, col)
                eval += bonus

                self.board[row][col] = 0

                if eval > best_value:
                    best_value = eval
                    best_move = (row, col)
                    if best_value >= 10:
                        break
        return best_move
    
    def get_available_places(self):
        available_places = []
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.board[row][col] == 0:
                    available_places.append((row, col))
        return available_places
        
    def minmax_alpha_beta(self, board, available_pieces, alpha, beta, is_maximizing, selected_piece, log=None):
        # DP 테이블 확인
        state_key = hash(np.array(board).tobytes()) ^ hash(str(selected_piece))
        if state_key in DP_table:
            return DP_table[state_key]

        # 승리 체크
        if selected_piece is None and log:
            row, col = log
            if check_win(board, row, col):
                # 보너스 적용
                bonus = get_position_bonus(row, col) + get_fork_bonus(board, get(board, row, col), row, col)
                return (10 if is_maximizing else -10) + bonus

        # 게임 종료 체크 
        if not available_pieces:
            return 0

        # 초기화
        best_eval = float('-inf') if is_maximizing else float('inf')
        update_func = max if is_maximizing else min

        if selected_piece is None:
            # 말 선택 단계
            for piece in available_pieces:
                remaining_pieces = [p for p in available_pieces if p != piece]
                eval = self.minmax_alpha_beta(board, remaining_pieces, alpha, beta, not is_maximizing, piece)
                best_eval = update_func(best_eval, eval)
                
                if is_maximizing:
                    alpha = max(alpha, eval)
                    if alpha == 10:
                        break
                else:
                    beta = min(beta, eval)
                    if beta == -10:
                        break
                    
                if beta <= alpha:
                    break
                    
        else:
            # 말 배치 단계
            for row, col in product(range(4), range(4)):
                if board[row][col] == 0:
                    board[row][col] = get_piece_idx(selected_piece)
                    eval = self.minmax_alpha_beta(board, available_pieces, alpha, beta, is_maximizing, None, (row, col))
                    # 보너스 추가
                    bonus = get_position_bonus(row, col) + get_fork_bonus(board, selected_piece, row, col)
                    eval += bonus
                    board[row][col] = 0
                    best_eval = update_func(best_eval, eval)
                    
                    if is_maximizing:
                        alpha = max(alpha, eval)
                        if alpha == 10:
                            break
                    else:
                        beta = min(beta, eval)
                        if beta == -10:
                            break
                        
                    if beta <= alpha:
                        break

        # DP 테이블에 저장
        DP_table[state_key] = best_eval
        return best_eval

# ===============================
# MCTS (몬테카를로 트리 탐색)
# ===============================

class MCTS:
    "Monte Carlo 트리 탐색기. 먼저 트리를 rollout한 뒤 수를 선택."
    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # 각 노드의 총 보상
        self.N = defaultdict(int)  # 각 노드의 방문 횟수
        self.children = dict()     # 각 노드의 자식 노드
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "노드의 최적 자식 선택 (게임에서 수 선택)"
        if self.children[node] is None:
            raise ValueError("탐색되지 않은 노드에서는 선택할 수 없음")
                
        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # 방문하지 않은 수는 피함
            return self.Q[n] / self.N[n]  # 평균 보상

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "트리를 한 단계 더 rollout (한 번 학습)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # 노드가 미탐색 또는 터미널 노드
                return path

            unexplored = set(self.children[node]) - set(self.children.keys())
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path

            node = self._uct_select(node)

    def _expand(self, node):
        if node in self.children:
            return  # 이미 확장됨
        
        node_list = node.find_children()
        self.children[node] = []
        for current_node in node_list:
            if current_node not in self.children:
                self.children[node].append(current_node)

    def _simulate(self, node):
        "노드에서 임의 시뮬레이션을 끝까지 돌려 보상 반환"
        while True:
            if node.is_terminal():
                return node.reward()
            node = node.find_random_child()

    def _backpropagate(self, path, reward):
        "리프 노드의 보상을 조상 노드로 전달"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):
        "탐험과 활용을 균형 있게 자식 노드 선택"
        assert all(n in self.children for n in self.children[node])
        log_N_vertex = math.log(self.N[node])

        def uct(n):
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)

# ===============================
# MCTS 노드
# ===============================

class Node():
    def __init__(self, board):
        self.board_state = board
        self.children = []

    def find_children(self):
        result = []
        # 플레이어 순서를 고려하여 자식 노드의 상태를 전환
        if self.board_state.selected_piece is None:  # 말 선택 단계
            for piece in self.board_state.available_pieces:
                next_board = copy.deepcopy(self.board_state)
                next_board.select(piece)
                next_node = Node(next_board)
                result.append(next_node)
        else:  # 말 배치 단계
            for row, col in self.board_state.available_places:
                next_board = copy.deepcopy(self.board_state)
                next_board.place(row, col)
                next_node = Node(next_board)
                result.append(next_node)
        self.children = result
        return result

    def find_random_child(self):
        return random.choice(self.children)

    def is_terminal(self):
        return not self.children

    def reward(self):
            copy_board = copy.deepcopy(self.board_state)
            "터미널 노드라고 가정. 1=승리, 0=패배, .5=무승부"
            while not is_board_full(copy_board.available_places):
                if copy_board.selected_piece is None:  # 말 선택 단계
                    copy_board.random_select()
                else:  # 말 배치 단계
                    row, col = copy_board.get_random_place()
                    copy_board.place(row, col)
                    if check_win(copy_board.get_board(), row, col):
                        return copy_board.player == PLAYER
            return 0.5

    def __hash__(self):
        return hash(self.board_state)

    def __eq__(self, other):
        return self.board_state == other.board_state
    
    def __str__(self):
        return str(self.board_state)

# ===============================
# MCTS 보드
# ===============================

class Board:
    def __init__(self, board, player, selected_piece, available_places, available_pieces):        
        self.__board = copy.deepcopy(board)
        self.player = player
        self.selected_piece = selected_piece
        self.available_places = copy.deepcopy(available_places)
        self.available_pieces = copy.deepcopy(available_pieces)
        if selected_piece in self.available_pieces:
            self.available_pieces.remove(selected_piece)
    
    def get_board(self):
        return self.__board

    def random_select(self):
        if self.selected_piece is not None:
            raise TypeError(f"지금은 '말 배치' 단계입니다")
        selected_piece = random.choice(self.available_pieces)
        self.select(selected_piece)

    def select(self, piece):
        if piece not in self.available_pieces:
            raise ValueError(f"선택한 말 {piece}는 사용 불가")
        self.player = -self.player
        self.selected_piece = piece
        self.available_pieces.remove(piece)
    
    def get_random_place(self):
        if self.selected_piece is None:
            raise TypeError("지금은 '말 선택' 단계입니다")
        selected_place = random.choice(self.available_places)
        return selected_place[0], selected_place[1]
    
    def place(self, row, col):
        if self.selected_piece is None:
            raise TypeError("지금은 '말 선택' 단계입니다")
        self.__board[row][col] = get_piece_idx(self.selected_piece)
        self.available_places.remove((row, col))
        self.selected_piece = None

    def __getitem__(self, index):
        return self.__board[index]
    
    def __hash__(self):
        return hash(self.__board.tobytes()) ^ hash(self.selected_piece)

    def __eq__(self, other):
        if not isinstance(other, Board):
            return False
        return (
            all(
                self.__board[row][col] == other.__board[row][col]
                for row in range(BOARD_ROWS)
                for col in range(BOARD_COLS)
            ) and
            self.selected_piece == other.selected_piece 
        )
