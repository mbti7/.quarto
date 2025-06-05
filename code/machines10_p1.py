import numpy as np
import random
from collections import defaultdict
from itertools import product
import math
import copy

# Algorithms parameters
MCTS_ITERATIONS = 1500
SWITCH_POINT = 8

# Constants
BOARD_ROWS = 4
BOARD_COLS = 4
pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces

# Variables
PLAYER = 1
isFirst = False # P2인 경우 True로 바꿔주세요.

# DP table for minimax
DP_table = {}

def restart_game():
    global DP_table
    DP_table = {}

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
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
        board = Board(self.board, 
                      PLAYER, 
                      None, 
                      self.available_places, 
                      self.available_pieces)
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
                False, piece, None, 
                depth = 0, depth_limit = 5
            )
            # print(f"{get_piece_text(piece)}: {eval}")
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

        # 즉시 승리 가능한 위치 확인
        for row, col in self.available_places:
            if check_win_with_piece(board, selected_piece, row, col):
                return row, col
        
        # MCTS 실행
        for _ in range(MCTS_ITERATIONS):
            tree.do_rollout(node)

        best_node = tree.choose(node)
        
        # 최적의 위치 반환
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
                    True, None, (row, col),
                    depth = 0, depth_limit = 5
                )
                self.board[row][col] = 0

                # print(f"({row}, {col}): {eval}")
                if eval > best_value:
                    best_value = eval
                    best_move = (row, col)
                    if best_value == 10:
                        break
        return best_move
    
    def get_available_places(self):
        available_places = []
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.board[row][col] == 0:
                    available_places.append((row, col))
        return available_places
        
    def minmax_alpha_beta(self, board, available_pieces, alpha, beta, is_maximizing, selected_piece, log=None, depth=0, depth_limit=5):
        # Check DP table
        state_key = hash(np.array(board).tobytes()) ^ hash(selected_piece)
        if state_key in DP_table:
            return DP_table[state_key]

        # 승리 체크
        if selected_piece is None and log:
            row, col = log
            if check_win(board, row, col):
                return 10 if is_maximizing else -10

        # 게임 종료 체크 
        if not available_pieces or depth >= depth_limit:
            return self.heuristic_eval(board, available_pieces, selected_piece)

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

        # Save to DP table
        DP_table[state_key] = best_eval
        return best_eval

    def heuristic_eval(self, board, available_pieces, selected_piece):
        # 기본: 중앙 우선, 한 줄에 같은 속성 개수, 즉시 위협 등 점수화
        score = 0

        # 1. 중앙 4칸 점수
        centers = [(1, 1), (1, 2), (2, 1), (2, 2)]
        for (x, y) in centers:
            piece = get(board, x, y)
            if piece is not None:
                score += 0.3  # 중앙에 두어진 말 개당 0.3점

        # 2. 한 줄에 같은 속성 3개(즉, 한 수만 더 두면 승리) 보너스
        for row in range(4):
            line = [get(board, row, col) for col in range(4)]
            if sum(p is not None for p in line) == 3:
                score += 0.5
        for col in range(4):
            line = [get(board, row, col) for row in range(4)]
            if sum(p is not None for p in line) == 3:
                score += 0.5

        # 3. 즉시 패배 위험: 남은 자리 중 한 곳에 두면 상대가 바로 이김
        for (row, col) in product(range(4), range(4)):
            if board[row][col] == 0:
                for piece in available_pieces:
                    board[row][col] = get_piece_idx(piece)
                    if check_win(board, row, col):
                        score -= 0.8
                    board[row][col] = 0

        return score

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."
    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if self.children[node] is None:
            raise ValueError("Cannot choose from unexplored node")
                
        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
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
                # Node is either unexplored or terminal
                return path

            # Explore unexplored nodes
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path

            # Descend deeper using UCT
            node = self._uct_select(node)

    def _expand(self, node):
        if node in self.children:
            return  # already expanded
        
        deleted_nodes = 0
        node_list = node.find_children()
        self.children[node] = []
        for currnet_node in node_list:
            if currnet_node not in self.children:
                self.children[node].append(currnet_node)
            else:
                deleted_nodes += 1

    def _simulate(self, node):
        """휴리스틱 기반 시뮬레이션"""
        while True:
            if node.is_terminal():
                return node.reward()
            # 1순위: 즉시 승리
            win_child = node.find_winning_child()
            if win_child:
                node = win_child
                continue
            # 2순위: 즉시 차단(상대가 이길 수 있는 수 방지)
            block_child = node.find_blocking_child()
            if block_child:
                node = block_child
                continue
            # 3순위: 중앙이 남아 있으면 중앙
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
            # 4순위: 랜덤
            node = node.find_random_child()
            
    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)

# =========================================
# MCTS Node
# MCTS에서는 메모리에 저장하기 때문에 따로 저장합니다.
# Minimax에서는 메모리에 저장하지 않고, 백트래킹을 이용합니다.
# =========================================

class Node():
    def __init__(self, board):
        self.board_state = board
        self.children = []

    def find_children(self):
        result = []
        # 플레이어 순서를 고려하여 자식 노드의 상태를 전환
        if self.board_state.selected_piece is None: # Select_piece
            for piece in self.board_state.available_pieces:
                next_board = copy.deepcopy(self.board_state)
                next_board.select(piece)
                next_node = Node(next_board)
                result.append(next_node)

        else: # Place_piece
            for row, col in self.board_state.available_places:
                next_board = copy.deepcopy(self.board_state)
                next_board.place(row, col)
                # last_move 정보를 board_state에 기록
                next_board.last_move = (row, col)
                next_node = Node(next_board)
                result.append(next_node)

        self.children = result
        return result

    def find_winning_child(self):
        """내가 지금 말을 놓으면 즉시 승리하는 자식 노드가 있으면 반환"""
        for child in self.find_children():
            if child.is_placing_state() and child.just_won():
                return child
        return None

    def find_blocking_child(self):
        """상대가 다음 수에 이길 수 있으면 그걸 막는 자식 노드가 있으면 반환"""
        # 자식 노드 중, 상대가 이기는 수를 막는 수 찾기
        for child in self.find_children():
            if child.is_placing_state() and child.blocks_opponent_win():
                return child
        return None

    def is_placing_state(self):
        # 말을 놓는(배치) 단계인지
        return self.board_state.selected_piece is not None

    def just_won(self):
        # 마지막 수가 승리인지 체크 (check_win 활용)
        last_move = self.board_state.last_move  # 필요시 last_move 추가
        if last_move is not None:
            board = self.board_state.get_board()
            return check_win(board, *last_move)
        return False

    def blocks_opponent_win(self):
        """상대가 다음 수에 즉시 이길 수 있는 상황을 방지하는지"""
        # 내 수를 두고 나서, 상대에게 남은 모든 수에 대해 check_win 시뮬레이션
        board = self.board_state.get_board()
        for (row, col) in self.board_state.available_places:
            tmp_board = copy.deepcopy(board)
            tmp_board[row][col] = get_piece_idx(self.board_state.selected_piece)
            if check_win(tmp_board, row, col):
                return False  # 상대에게 승리 수를 허용
        return True
        

    def find_random_child(self):
        return random.choice(self.children)

    def is_terminal(self):
        return not self.children

    def reward(self):
        copy_board = copy.deepcopy(self.board_state)
        "Assumes self is terminal node. 1=win, 0=loss, .5=draw"

        while not is_board_full(copy_board.available_places):
            if copy_board.selected_piece is None: # Select_piece
                copy_board.random_select()

            else: # Place_piece
                row, col = copy_board.get_random_place()
                copy_board.place(row, col)
                if check_win(copy_board.get_board(), row, col):
                    return copy_board.player == PLAYER

        return 0.5

    # =========================================
    # Special Methods
    # =========================================

    def __hash__(self):
        return hash(self.board_state)

    def __eq__(self, other):
        return self.board_state == other.board_state
    
    def __str__(self):
        return str(self.board_state)
    

# =========================================
# MCTS Board
# =========================================

class Board:
    def __init__(self, board, player, selected_piece, available_places, available_pieces):        
        self.__board = copy.deepcopy(board)
        self.player = player
        self.selected_piece = selected_piece
        self.available_places = copy.deepcopy(available_places)
        self.available_pieces = copy.deepcopy(available_pieces)
        self.last_move = None
        if selected_piece in self.available_pieces:
            self.available_pieces.remove(selected_piece)
    
    def get_board(self):
        return self.__board

    def random_select(self):
        if self.selected_piece is not None:
            raise TypeError(f"Now is 'place_piece' state")

        selected_piece = random.choice(self.available_pieces)

        self.select(selected_piece)

    def select(self, piece):
        if piece not in self.available_pieces:
            raise ValueError(f"The selected piece {piece} is not available")

        self.player = -self.player
        self.selected_piece = piece
        self.available_pieces.remove(piece)
    
    def get_random_place(self):
        if self.selected_piece is None:
            raise TypeError("Now is 'select_piece' state")
        
        selected_place = random.choice(self.available_places)
        return selected_place[0], selected_place[1]
    
    def place(self, row, col):
        if self.selected_piece is None:
            raise TypeError("Now is 'select_piece' state")

        self.__board[row][col] = get_piece_idx(self.selected_piece)
        self.available_places.remove((row, col))
        self.selected_piece = None
        self.last_move = (row, col)

    # =========================================
    # Special Methods
    # =========================================

    def __str__(self):
        # Convert the board into a readable string
        board_str = ''
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                piece = get(self.get_board(), row, col)
                board_str += f"{get_piece_text(piece)} "
            board_str += '\n'
        return f"\nPlayer: {self.player}, Selected_piece: {get_piece_text(self.selected_piece)}\n{board_str}\n"
        
    def __getitem__(self, index):
        return self.__board[index]
    
    def __hash__(self):
        return hash(self.__board.tobytes()) ^ hash(self.selected_piece)

    def __eq__(self, other):
        # Equality check for hash compatibility
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

# =========================================
# General Functions
# =========================================

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

def check_win_with_piece(board, piece, x, y):
    # Place the piece temporarily
    if piece not in pieces:
        raise ValueError(f"Piece {piece} is not available")
    board[x][y] = get_piece_idx(piece)
    flag = check_win(board, x, y)
    board[x][y] = 0
    return flag

def get_piece_text(piece):
    if piece is None:
        return "...."
    else:
        return f"{'I' if piece[0] == 0 else 'E'}{'N' if piece[1] == 0 else 'S'}{'T' if piece[2] == 0 else 'F'}{'P' if piece[3] == 0 else 'J'}"

def print_piece(piece):
    print(get_piece_text(piece))

def print_board(board):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            print(get_piece_text(get(board, row, col)), end=" ")
        print()
    print("--------------------------------")

def check_win(board, x, y):
    def check_equal_attributes(pieces):
        return any(all(piece[i] == pieces[0][i] for piece in pieces) for i in range(4))

    def check_line(pieces):
        return None not in pieces and check_equal_attributes(pieces)

    def get_2x2_pieces(i, j):
        return [
            get(board, i, j), get(board, i, j + 1),
            get(board, i + 1, j), get(board, i + 1, j + 1)
        ]

    # 가로줄 확인
    row_pieces = [get(board, x, j) for j in range(BOARD_COLS)]
    if check_line(row_pieces):
        return True

    # 세로줄 확인
    col_pieces = [get(board, i, y) for i in range(BOARD_ROWS)]
    if check_line(col_pieces):
        return True

    # 주대각선 확인 (좌상단 -> 우하단)
    if x == y:
        diag_pieces = [get(board, i, i) for i in range(BOARD_ROWS)]
        if check_line(diag_pieces):
            return True

    # 부대각선 확인 (우상단 -> 좌하단)
    if x + y == BOARD_ROWS - 1:
        anti_diag_pieces = [get(board, i, BOARD_ROWS - 1 - i) for i in range(BOARD_ROWS)]
        if check_line(anti_diag_pieces):
            return True

    # 2x2 영역 확인
    for i in range(max(0, x - 1), min(BOARD_ROWS - 2, x) + 1):
        for j in range(max(0, y - 1), min(BOARD_COLS - 2, y) + 1):
            if check_line(get_2x2_pieces(i, j)):
                return True

    return False
