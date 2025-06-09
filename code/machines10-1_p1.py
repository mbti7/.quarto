import numpy as np
import random
from collections import defaultdict
from itertools import product
import math
import copy

# ==============================
# 기본 설정 / Constants
# ==============================

MCTS_ITERATIONS = 1500  # MCTS 반복 횟수
SWITCH_POINT = 8        # Minimax 전환 시점

BOARD_ROWS = 4
BOARD_COLS = 4
pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # 총 16개의 조각

PLAYER = 1
isFirst = False  # P1이므로 상대(P2)가 먼저 시작

DP_table = {}  # Minimax용 DP 메모이제이션 테이블

def restart_game():
    global DP_table
    DP_table = {}

# ==============================
# 플레이어 P1 클래스
# ==============================

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = pieces[:]
        self.board = board  # 0: 빈칸 / 1~16: 조각 인덱스
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

        # 상대가 이길 수 있는 말은은 제외
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
                False, piece, None, 0, 5
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
        if len(self.available_places) >= 15:
            centers = [(1, 1), (1, 2), (2, 1), (2, 2)]
            center_candidates = [pos for pos in centers if self.board[pos[0]][pos[1]] == 0]
            if center_candidates:
                return random.choice(center_candidates)

        tree = MCTS()
        board = Board(self.board, PLAYER, selected_piece, self.available_places, self.available_pieces)
        node = Node(board)

        for row, col in self.available_places:
            if check_win_with_piece(board, selected_piece, row, col):
                return row, col

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
                    True, None, (row, col), 0, 5
                )
                self.board[row][col] = 0

                if eval > best_value:
                    best_value = eval
                    best_move = (row, col)
                    if best_value == 10:
                        break
        return best_move

    def get_available_places(self):
        return [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]

       # MinMax함수의 구성성   
    def minmax_alpha_beta(self, board, available_pieces, alpha, beta, is_maximizing, selected_piece, log=None, depth=0, depth_limit=5):
        state_key = hash(np.array(board).tobytes()) ^ hash(selected_piece)
        if state_key in DP_table:
            return DP_table[state_key]

        if selected_piece is None and log:
            row, col = log
            if check_win(board, row, col):
                return 10 if is_maximizing else -10

        if not available_pieces or depth >= depth_limit:
            return self.heuristic_eval(board, available_pieces, selected_piece)

        best_eval = float('-inf') if is_maximizing else float('inf')
        update_func = max if is_maximizing else min

        if selected_piece is None:
            for piece in available_pieces:
                next_pieces = [p for p in available_pieces if p != piece]
                eval = self.minmax_alpha_beta(board, next_pieces, alpha, beta, not is_maximizing, piece, None, depth+1, depth_limit)
                best_eval = update_func(best_eval, eval)
                if is_maximizing:
                    alpha = max(alpha, eval)
                    if alpha >= 10:
                        break
                else:
                    beta = min(beta, eval)
                    if beta <= -10:
                        break
                if beta <= alpha:
                    break
        else:
            for row, col in product(range(4), range(4)):
                if board[row][col] == 0:
                    board[row][col] = get_piece_idx(selected_piece)
                    eval = self.minmax_alpha_beta(board, available_pieces, alpha, beta, is_maximizing, None, (row, col), depth+1, depth_limit)
                    board[row][col] = 0
                    best_eval = update_func(best_eval, eval)
                    if is_maximizing:
                        alpha = max(alpha, eval)
                        if alpha >= 10:
                            break
                    else:
                        beta = min(beta, eval)
                        if beta <= -10:
                            break
                    if beta <= alpha:
                        break

        DP_table[state_key] = best_eval
        return best_eval

    def heuristic_eval(self, board, available_pieces, selected_piece):
        score = 0
        centers = [(1, 1), (1, 2), (2, 1), (2, 2)]
        for x, y in centers:
            if get(board, x, y) is not None:
                score += 0.3

        for i in range(4):
            row = [get(board, i, j) for j in range(4)]
            if sum(p is not None for p in row) == 3:
                score += 0.5
            col = [get(board, j, i) for j in range(4)]
            if sum(p is not None for p in col) == 3:
                score += 0.5

        for r, c in product(range(4), range(4)):
            if board[r][c] == 0:
                for piece in available_pieces:
                    board[r][c] = get_piece_idx(piece)
                    if check_win(board, r, c):
                        score -= 0.8
                    board[r][c] = 0
        return score

class MCTS:
    """MCTS (Monte Carlo Tree Search) 클래스 – 트리 탐색을 통해 최적의 수를 선택합니다."""

    def __init__(self, exploration_weight=1):
        # Q: 각 노드의 누적 보상값 (Total reward of each node)
        # N: 각 노드의 방문 횟수 (Visit count of each node)
        # children: 각 노드의 자식 노드 목록
        self.Q = defaultdict(int)
        self.N = defaultdict(int)
        self.children = dict()
        self.exploration_weight = exploration_weight  # 탐색 계수 (UCT 수식 내 탐험 성향 조절)

    def choose(self, node):
        """루트 노드에서 가장 좋은 자식 노드를 선택합니다 (탐색 완료 후 move 선택)."""
        if self.children[node] is None:
            raise ValueError("아직 탐색되지 않은 노드에서는 선택할 수 없습니다.")
        
        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # 한 번도 시도되지 않은 노드는 제외
            return self.Q[n] / self.N[n]  # 평균 보상값 (평균 승률)

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        """MCTS 1회 수행: selection → expansion → simulation → backpropagation."""
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        """Selection 단계: 탐색 트리에서 UCT 기반으로 경로를 선택합니다."""
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # 확장되지 않은 노드이거나 터미널 노드일 경우 종료
                return path

            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path

            # UCT 값을 기반으로 가장 우수한 자식 노드 선택
            node = self._uct_select(node)

    def _expand(self, node):
        """Expansion 단계: 현재 노드에서 가능한 자식 노드를 생성합니다."""
        if node in self.children:
            return  # 이미 확장된 노드이면 무시

        node_list = node.find_children()
        self.children[node] = []
        for current_node in node_list:
            if current_node not in self.children:
                self.children[node].append(current_node)

    def _simulate(self, node):
        """Simulation 단계: 휴리스틱 기반의 간이 게임 시뮬레이션을 수행합니다."""
        while True:
            if node.is_terminal():
                return node.reward()

            # 1순위: 즉시 승리 가능한 자식
            win_child = node.find_winning_child()
            if win_child:
                node = win_child
                continue

            # 2순위: 상대 승리 차단
            block_child = node.find_blocking_child()
            if block_child:
                node = block_child
                continue

            # 3순위: 중앙 자리에 둘 수 있는 경우
            center = (1, 1), (1, 2), (2, 1), (2, 2)
            central_child = None
            for child in node.find_children():
                if getattr(child, "board_state", None) and \
                   getattr(child.board_state, "last_move", None) in center:
                    central_child = child
                    break
            if central_child:
                node = central_child
                continue

            # 4순위: 랜덤 자식 선택
            node = node.find_random_child()

    def _backpropagate(self, path, reward):
        """Backpropagation 단계: 시뮬레이션 결과를 경로를 따라 역전파합니다."""
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):
        """UCT(Upper Confidence Bound for Trees) 공식 기반 자식 노드 선택"""
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            # UCT 수식 = 평균 보상 + 탐험 보너스
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)

class Node():
    """
    MCTS 트리의 노드 클래스.
    각 노드는 게임 상태(Board)를 포함하며,
    가능한 자식 노드 탐색 및 평가 기능을 제공합니다.
    """

    def __init__(self, board):
        self.board_state = board  # 현재 게임 상태
        self.children = []        # 자식 노드 목록

    def find_children(self):
        """
        현재 상태에서 가능한 모든 다음 수(child node)를 생성하여 반환.
        - 선택 단계(select_piece)면 가능한 모든 조각 선택
        - 배치 단계(place_piece)면 가능한 모든 위치에 조각 배치
        """
        result = []

        if self.board_state.selected_piece is None:  # 조각 선택 단계
            for piece in self.board_state.available_pieces:
                next_board = copy.deepcopy(self.board_state)
                next_board.select(piece)
                result.append(Node(next_board))

        else:  # 조각 배치 단계
            for row, col in self.board_state.available_places:
                next_board = copy.deepcopy(self.board_state)
                next_board.place(row, col)
                next_board.last_move = (row, col)  # 마지막 수 기록
                result.append(Node(next_board))

        self.children = result
        return result

    def find_winning_child(self):
        """
        즉시 승리 가능한 수가 있는 자식 노드가 있다면 반환.
        """
        for child in self.find_children():
            if child.is_placing_state() and child.just_won():
                return child
        return None

    def find_blocking_child(self):
        """
        상대가 다음 수에 이길 수 있는 경우, 그것을 차단할 수 있는 자식 노드 반환.
        """
        for child in self.find_children():
            if child.is_placing_state() and child.blocks_opponent_win():
                return child
        return None

    def is_placing_state(self):
        """
        현재 상태가 조각을 배치해야 하는 상황인지 여부 반환.
        """
        return self.board_state.selected_piece is not None

    def just_won(self):
        """
        마지막 수가 승리 조건을 만족했는지 판단.
        """
        last_move = self.board_state.last_move
        if last_move is not None:
            board = self.board_state.get_board()
            return check_win(board, *last_move)
        return False

    def blocks_opponent_win(self):
        """
        상대방이 다음 수에 이길 수 있는지를 사전에 막았는지 확인.
        """
        board = self.board_state.get_board()
        for (row, col) in self.board_state.available_places:
            tmp_board = copy.deepcopy(board)
            tmp_board[row][col] = get_piece_idx(self.board_state.selected_piece)
            if check_win(tmp_board, row, col):
                return False  # 막지 못함
        return True  # 모두 차단 성공

    def find_random_child(self):
        """
        자식 노드 중 임의로 하나를 반환.
        (시뮬레이션에서 사용)
        """
        return random.choice(self.children)

    def is_terminal(self):
        """
        더 이상 자식 노드가 없는 종료 상태인지 여부 반환.
        """
        return not self.children

    def reward(self):
        """
        현재 노드가 종료 상태라 가정하고, 남은 수를 무작위로 시뮬레이션하여 보상을 반환.
        - 1: 승리
        - 0: 패배
        - 0.5: 무승부
        """
        copy_board = copy.deepcopy(self.board_state)

        while not is_board_full(copy_board.available_places):
            if copy_board.selected_piece is None:
                copy_board.random_select()
            else:
                row, col = copy_board.get_random_place()
                copy_board.place(row, col)
                if check_win(copy_board.get_board(), row, col):
                    return copy_board.player == PLAYER

        return 0.5

    # =========================================
    # Python 내장 함수 오버라이딩 (hash, equals, string)
    # =========================================

    def __hash__(self):
        return hash(self.board_state)

    def __eq__(self, other):
        return self.board_state == other.board_state

    def __str__(self):
        return str(self.board_state)

class Board:
    """
    게임 보드 상태를 표현하는 클래스.
    - 현재 보드 배열
    - 현재 플레이어
    - 선택된 조각
    - 배치 가능한 위치 및 선택 가능한 조각 목록
    - 마지막 수 위치 기록
    """

    def __init__(self, board, player, selected_piece, available_places, available_pieces):        
        self.__board = copy.deepcopy(board)         # 실제 게임 보드 (4x4)
        self.player = player                         # 현재 플레이어 (1 또는 -1)
        self.selected_piece = selected_piece         # 현재 선택된 조각 (배치 전 상태일 수 있음)
        self.available_places = copy.deepcopy(available_places)  # 말이 놓일 수 있는 위치 목록
        self.available_pieces = copy.deepcopy(available_pieces)  # 아직 선택되지 않은 조각 목록
        self.last_move = None                        # 마지막 수의 좌표 (row, col)
        
        # 선택된 조각은 available 목록에서 제거
        if selected_piece in self.available_pieces:
            self.available_pieces.remove(selected_piece)

    def get_board(self):
        """현재 보드 상태 반환 (내부 복사본)."""
        return self.__board

    def random_select(self):
        """무작위로 조각을 선택 (선택 상태일 때만 가능)."""
        if self.selected_piece is not None:
            raise TypeError("현재는 '배치' 단계입니다. (Already in place_piece state)")

        selected_piece = random.choice(self.available_pieces)
        self.select(selected_piece)

    def select(self, piece):
        """지정된 조각을 선택하고 플레이어 턴을 변경합니다."""
        if piece not in self.available_pieces:
            raise ValueError(f"선택할 수 없는 조각입니다: {piece}")

        self.player = -self.player
        self.selected_piece = piece
        self.available_pieces.remove(piece)

    def get_random_place(self):
        """무작위로 배치 가능한 위치를 반환합니다."""
        if self.selected_piece is None:
            raise TypeError("현재는 '선택' 단계입니다. (Already in select_piece state)")

        selected_place = random.choice(self.available_places)
        return selected_place[0], selected_place[1]

    def place(self, row, col):
        """지정된 위치에 조각을 배치합니다."""
        if self.selected_piece is None:
            raise TypeError("현재는 '선택' 단계입니다. (No selected piece)")

        self.__board[row][col] = get_piece_idx(self.selected_piece)  # 말 배치
        self.available_places.remove((row, col))                    # 자리 제거
        self.selected_piece = None                                  # 조각 배치 완료
        self.last_move = (row, col)                                 # 마지막 수 기록

    # =========================================
    # Python 특수 메서드 오버라이딩
    # =========================================

    def __str__(self):
        """보드 상태를 문자열로 반환 (디버깅 및 출력용)."""
        board_str = ''
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                piece = get(self.get_board(), row, col)
                board_str += f"{get_piece_text(piece)} "
            board_str += '\n'
        return f"\nPlayer: {self.player}, Selected_piece: {get_piece_text(self.selected_piece)}\n{board_str}\n"

    def __getitem__(self, index):
        """인덱스로 보드 행 접근 가능하게 만듦."""
        return self.__board[index]
    
    def __hash__(self):
        """보드 상태 해싱 (MCTS 노드 비교/캐시에 사용)."""
        return hash(self.__board.tobytes()) ^ hash(self.selected_piece)

    def __eq__(self, other):
        """보드 동등 비교 (내용과 선택된 조각 동일 시 동일한 상태)."""
        if not isinstance(other, Board):
            return False

        return (
            all(
                self.__board[row][col] == other.__board[row][col]
                for row in range(BOARD_ROWS)
                for col in range(BOARD_COLS)
            ) and self.selected_piece == other.selected_piece
        )

def get(board, x, y):
    """
    보드에서 (x, y) 위치의 조각을 반환.
    - 0이면 None 반환
    - 1~16이면 해당 인덱스에 대응하는 조각 반환
    """
    piece_idx = board[x][y] - 1
    return None if piece_idx == -1 else pieces[piece_idx]

def get_piece_idx(piece):
    """
    조각의 인덱스를 반환 (보드에 저장될 정수값).
    - pieces 리스트의 인덱스 + 1
    """
    return pieces.index(piece) + 1

def is_board_full(available_places):
    """
    남은 배치 가능 위치가 없으면 True (보드가 가득 참).
    """
    return not available_places

def check_win_with_piece(board, piece, x, y):
    """
    주어진 위치에 조각을 임시로 배치한 뒤 승리 여부를 판단하고 원상복구.
    """
    if piece not in pieces:
        raise ValueError(f"잘못된 조각입니다: {piece}")
    
    board[x][y] = get_piece_idx(piece)
    flag = check_win(board, x, y)
    board[x][y] = 0
    return flag

def get_piece_text(piece):
    """
    조각 속성을 문자열로 변환.
    예: (0,0,1,1) → 'INTJ'
    """
    if piece is None:
        return "...."
    return f"{'I' if piece[0] == 0 else 'E'}" + \
           f"{'N' if piece[1] == 0 else 'S'}" + \
           f"{'T' if piece[2] == 0 else 'F'}" + \
           f"{'P' if piece[3] == 0 else 'J'}"

def print_piece(piece):
    """
    조각을 문자열로 출력.
    """
    print(get_piece_text(piece))

def print_board(board):
    """
    전체 보드 상태를 보기 좋게 출력.
    """
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            print(get_piece_text(get(board, row, col)), end=" ")
        print()
    print("--------------------------------")

def check_win(board, x, y):
    """
    주어진 위치 (x, y)를 기준으로 승리 조건 만족 여부를 검사.
    승리 조건:
    - 가로, 세로, 대각선, 2x2 블록 중 하나라도
      동일 속성 4개가 연속된 경우
    """

    def check_equal_attributes(pieces):
        # 4개의 조각이 동일한 속성을 하나 이상 공유하는지 확인
        return any(all(piece[i] == pieces[0][i] for piece in pieces) for i in range(4))

    def check_line(pieces):
        # None 없이 4개 모두 존재하고, 하나의 속성이 같다면 승리
        return None not in pieces and check_equal_attributes(pieces)

    def get_2x2_pieces(i, j):
        # 2x2 영역에 있는 4개의 조각 반환
        return [
            get(board, i, j), get(board, i, j + 1),
            get(board, i + 1, j), get(board, i + 1, j + 1)
        ]

    # 1. 가로줄 검사
    if check_line([get(board, x, j) for j in range(BOARD_COLS)]):
        return True

    # 2. 세로줄 검사
    if check_line([get(board, i, y) for i in range(BOARD_ROWS)]):
        return True

    # 3. 주대각선 검사 (좌상단 → 우하단)
    if x == y:
        if check_line([get(board, i, i) for i in range(BOARD_ROWS)]):
            return True

    # 4. 부대각선 검사 (우상단 → 좌하단)
    if x + y == BOARD_ROWS - 1:
        if check_line([get(board, i, BOARD_ROWS - 1 - i) for i in range(BOARD_ROWS)]):
            return True

    # 5. 2x2 블록 검사
    for i in range(max(0, x - 1), min(BOARD_ROWS - 2, x) + 1):
        for j in range(max(0, y - 1), min(BOARD_COLS - 2, y) + 1):
            if check_line(get_2x2_pieces(i, j)):
                return True

    return False
