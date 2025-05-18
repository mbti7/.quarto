import numpy as np
import random
from itertools import product

import time
import copy
BOARD_ROWS = 4
BOARD_COLS = 4
EXPLORATION_WEIGHT = 1.5
WEIGHT_CHANGE = 0.04 # 뒤로갈 수록 가중치를 줄여 집중탐색할거임

CALC_TIME = [1,1,1,1,2,2,2,2,2,2,2,2,2,14,21,35,39,44,49,54,60,59,49,44,35,35,31,1,1,1,1,1]
# CALC_TIME = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces

# 코드설명
# 
# 트리는 P1,P2 class의 정적변수로서, 게임이 진행되는 동안 지워지지 않습니다.
# 트리를 구성하는 노드는 2종류 (piece선택노드, location선택 노드) 있습니다.
# 각 노드는 나 or 상대의 것임을 mine이라는 변수가 저장하고 있습니다.
# 
# 매 턴마다 몬테카를로 알고리즘을 진행합니다. 이기면 +1, 지면 -1, 비기면 0입니다.
# 가끔 트리가 완성되는 경우가 있는데, 이 경우에는 완성된 트리의 루트노드의 done변수가 그를 반영합니다.
# done은 이기면 1, 지면 -1, 비기면 0.0001을 지니게 됩니다. simulate을 돌릴지, best_child인지 검사할때 중요하게 쓰입니다.
# done은 한번 갱신되면 부모노드의 done에도 영향을 줍니다.
# 
# ucb의 상수값은 1.5에서 천천히 내려와 범위탐색에서 집중탐색으로 전환합니다. (그러길 바라..)
# ucb의 값을 계산할때, x값(시뮬레이션 평균값)은 절댓값을 취했습니다.

# 코드 순서
# 
# while (시간 조건):
#   head(루트노드) 갱신
#   몬테카를로
#   best_child 선택

# 2진수의 piece를 mbti문자열로 반환 (비중요)
def toString_mbti(tuple):
    str = ''
    if (tuple[0] == 0):
        str += 'I'
    else:
        str += 'E'
    if (tuple[1] == 0):
        str += 'N'
    else:
        str += 'S'
    if (tuple[2] == 0):
        str += 'T'
    else:
        str += 'F'
    if (tuple[3] == 0):
        str += 'P'
    else:
        str += 'J'
    return str

# 보드출력 (디버깅용)
def print_board(board):
    for row in board:
        list = []
        for col in row:
            if (col == 0):
                list.append('    ')
            else:
                list.append(toString_mbti(pieces[col-1]))
        print(list)

# 노드 출력 (디버깅용)
def print_child_data(node):
    for child in node.childNodes:
        print("..........................",child.depth)
        print_board(child.board)
        print('x:',child.x)
        print('n: ',child.n)
        print('done:', child.done)
        print(child.data)

# 빈 보드 하나 생성
def new_board():
    return np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)

# 모든 piece가 들은 배열 생성
def new_pieces():
    return [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]

# childNodes배열의 child중, 특정 data(selected_piece or place_location)를 가진놈을 반환
def detect_child(childNodes, data):
    for child in childNodes:
        if (child.data == data):
            return child
    return None

# 원본 보드와 갱신된 보드를 가져와 새로운 piece가 어디에 두어졌는지 계산 
# 상대의 턴이 지난 후 계산할 필요가 있음
def detect_placed_location(origin_board, new_board):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if origin_board[row][col] != new_board[row][col]:
                return (row, col)
    return None

# place가능한 location반환
def detect_able_locs(board):
    return [(row, col) for row, col in product(range(4), range(4)) if board[row][col]==0]    

# 승리조건 (비중요)
def check_line(line):
    if 0 in line:
        return False  # Incomplete line
    characteristics = np.array([pieces[piece_idx - 1] for piece_idx in line])
    for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
        if len(set(characteristics[:, i])) == 1:  # All share the same characteristic
            return True
    return False

# 승리조건 (비중요)
def check_2x2_subgrid_win(board):
    for r in range(BOARD_ROWS - 1):
        for c in range(BOARD_COLS - 1):
            subgrid = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
            if 0 not in subgrid:  # All cells must be filled
                characteristics = [pieces[idx - 1] for idx in subgrid]
                for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
                    if len(set(char[i] for char in characteristics)) == 1:  # All share the same characteristic
                        return True
    return False

# 승리조건 (비중요)
def check_win(board):
    # Check rows, columns, and diagonals
    for col in range(BOARD_COLS):
        if check_line([board[row][col] for row in range(BOARD_ROWS)]):
            return True
    
    for row in range(BOARD_ROWS):
        if check_line([board[row][col] for col in range(BOARD_COLS)]):
            return True
        
    if check_line([board[i][i] for i in range(BOARD_ROWS)]) or check_line([board[i][BOARD_ROWS - i - 1] for i in range(BOARD_ROWS)]):
        return True

    # Check 2x2 sub-grids
    if check_2x2_subgrid_win(board):
        return True
    
    return False

# 나 or 상대가 선택한 piece 정보 포함
class select_Node():
    def __init__(self, parent, piece, able_pieces, mine, depth = 0):
        self.parent = parent            # 부모
        self.data = piece               # 선택한 piece
        self.able_pieces = able_pieces  # 남은 piece들
        self.mine = mine                # true면 나, flase면 상대
        self.depth = depth              # 노드의 깊이 0~32

        self.board = parent.board       # 보드
        self.childNodes = []            # 자식노드(class: place_Node)
        self.x = 0.0                    # 시뮬레이션 평균값
        self.n = 0                      # 방문 수
        self.done = 0                  # 승패판단이 명확해짐 (1: 내가 이김 0.0001: 무승부 -1: 상대가 이길 수 있음)

    #childNodes확장 (이미 있는것 재외)
    def expand(self):
        parent = self.parent

        for row,col in detect_able_locs(parent.board):
            if (not detect_child(self.childNodes,(row,col))):
                newBoard = copy.deepcopy(parent.board)              # 새로만든 보드
                newBoard[row][col] = pieces.index(self.data) + 1    # 보드 임의의 곳에 부모의 piece두는 것
                newAblePiece = copy.deepcopy(self.able_pieces)      # piece를 두고 남은 piece들

                newNode = place_Node(self, newBoard, (row,col), newAblePiece, not self.mine, self.depth+1) # depth++
                self.childNodes.append(newNode)
                
# 나 or 상대가 선택한 location 정보 포함
class place_Node():
    def __init__(self, parent, board, location, able_pieces, mine, depth = 0):
        self.parent = parent            # 부모
        self.board = board              # 보드
        self.data = location            # 선택한 location
        self.able_pieces = able_pieces  # 남은 piece들
        self.mine = mine                # true면 나, flase면 상대
        self.depth = depth              # 노드의 깊이 0~32

        self.childNodes = []            # 자식노드(class: place_Node)
        self.x = 0.0                    # 시뮬레이션 평균값
        self.n = 0                      # 방문 수
        self.done = 0                  # 승패판단이 명확해짐 (1: 내가 이김 0.0001: 무승부 -1: 상대가 이길 수 있음)
    
    #childNodes확장 (이미 있는것 재외)
    def expand(self):
        for piece in self.able_pieces:
            if (not detect_child(self.childNodes, piece)):
                newAblePiece = copy.deepcopy(self.able_pieces)  
                newAblePiece.remove(piece) # piece를 둠으로서 남은 piece들

                newNode = select_Node(self, piece, newAblePiece, self.mine, self.depth+1) # depth++
                self.childNodes.append(newNode)

# p2 *** player에 따라 다름! ***
class P2():
    head = place_Node(None, new_board() , None, new_pieces(), True) # root_node
    # P2가 루트노드의 주인이기에, mine 인자값으로 False를 줌

    current_location = None # 각각, 상대가 뭘 선택했는지 등을 계산하기 위해 정적으로 선언한 변수들. (비중요)
    current_board = new_board()
    current_piece = None
    def __init__(self, board, available_pieces):  # 생성자
        # self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces

        # 게임이 다시 시작된다면 reset
        if len(available_pieces) == 16:
            P2.reset()

    # 상대가 둘 piece 선정
    def select_piece(self):
        if (P2.head.parent != None):
            head_set([P2.current_location]) # 헤드 수정 (첫부분은 다르게 시행)
        selected_piece = monte_carlo(P2.head) # 몬테카를로 -> select
        print("P2: selected ", selected_piece, toString_mbti(selected_piece))
        P2.current_piece = selected_piece
        P2.current_board = copy.deepcopy(self.board)
        return selected_piece

    # 내가 둘 location 선정
    def place_piece(self, selected_piece):
        placed_loc = detect_placed_location(P2.current_board, self.board)
        head_set([P2.current_piece, placed_loc, selected_piece]) # 헤드 수정 
        position = monte_carlo(P2.head) # 몬테카를로 -> select
        print("P2: placed at", position)
        P2.current_location = position
        return position
    
    def reset():
        P2.head = place_Node(None, new_board() , None, new_pieces(), True)
    
# monte carlo 트리의 head를 갱신해줌. (게임이 진행됨에 따라 필요없는 부모노드를 무시) *** player에 따라 다름! ***
def head_set(list):
    for data in list:
        P2.head.expand()
        P2.head = detect_child(P2.head.childNodes, data)
        
        if (not P2.head):
            print(data,"실패")

# monte carlo 진행 (iterate할 노드 선정 -> simulate -> backpropagate)
def monte_carlo(root):
    begin = time.time() # 시간측정

    if (not root.done): # 완성된 tree가 아님
        while time.time() < begin + CALC_TIME[root.depth]: # 범위에 맞는 시간만큼

            if root.done: # tree가 완성됨! 명확히 자신이 이길지 질지 예측 가능
                print('P2: all possible cases calculated.')
                if(root.done == 1):
                    print('P2: sure win!')
                else:
                    print('P2: might lose.')
                break

            node = root
            while node.childNodes: # simulate할 노드 선택
                node = select_to_roll_child(node)

            x = simulate_game(node) # simulate
            backpropagate(node, x) # backpropagate

    best_child = select_best_child(root) # 최종 선택
    return best_child.data

# simulate할 노드 선정 (ucb1)
def select_to_roll_child(node):
    def ucb1_score(child):
        dont_care = 0
        if child.n == 0:
            return float('inf')  # 방문한 적 없는 자식노드 최우선
        
        if child.done == 1:
            dont_care = 999 # 이기는 child는 볼 필요 없음
            
        exploitation = abs(child.x) # 중요! 절댓값!
        exploration = (EXPLORATION_WEIGHT - WEIGHT_CHANGE*child.depth) * np.sqrt(np.log(node.n) / child.n)
        return exploitation + exploration - dont_care

    return max(node.childNodes, key=ucb1_score)

# head노드의 자식중 최선의 선택을 고름
def select_best_child(node):
    return max(node.childNodes, key=lambda child: child.x + child.done) # done을 고려함으로써 승패를 확정

# simualte   
def simulate_game(node):
    node.expand()
    
    # 이것저것 깊은복사해옴
    temp_board = copy.deepcopy(node.board)
    temp_able_pieces = copy.deepcopy(node.able_pieces)
    temp_able_locs = detect_able_locs(temp_board)
    current_mine = node.mine # 현재 바라보는 노드가 내것인가 상대것인가

    # 게임이 끝난 노드인지 확인
    if check_win(temp_board):
        if (current_mine):
            node.done = 1
        else:
            node.done = -1

    # 무승부
    if not len(temp_able_pieces):
        node.done = 0.0001

    #select노드였다면 초반에 추가 진행 + 1 (simulate하기에 부적합하게 생겼음)
    if (isinstance(node, select_Node)):
        current_mine = not current_mine
        row, col = random.choice(temp_able_locs)
        temp_able_locs.remove((row,col))
        temp_board[row][col] = pieces.index(node.data) + 1

    # 게임이 끝날때 까지 계속 simulate
    while not check_win(temp_board):
        if (not temp_able_pieces):
            return 0
        
        current_mine = not current_mine

        piece = random.choice(temp_able_pieces)
        row, col = random.choice(temp_able_locs)

        temp_able_pieces.remove(piece)
        temp_able_locs.remove((row,col))

        temp_board[row][col] = pieces.index(piece) + 1

    if (current_mine == node.mine):
        return 1 # 승리
    else:
        return -1 # 패배

# backpropagate하여 부모노드의 n, x등을 갱신
def backpropagate(node, x):
    temp_x = x

    while node is not None: # 위로 쭉
        if (not node.mine):
            temp_x = -x # min_max 개념으로, 상대방노드의 값은 반전시킬 필요가 있음.

        adder = 1 if isinstance(node, place_Node) == node.mine else -1
        # 나의 place_Node       ->  1
        # 나의 select_Node      -> -1
        # 상대의 place_Node     -> -1
        # 상대의 select_Node    ->  1

        # child가 완전한 트리가 됨에 따라 부모도 완전해졌는지 검사
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