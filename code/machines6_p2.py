import numpy as np
import random
from itertools import product

import time

    # 1) 중앙 선점
    #       가장 많이 쓰이는 전략
    #       첫 수(혹은 첫 두 수)는 중앙(2,2), (2,3), (3,2), (3,3) 네 칸을 선점하는 것이 유리
    #       이유: 중앙은 가로, 세로, 대각선 3개에 모두 관여 → 승리 루트가 많아짐
    # 2) 코너 활용
    #       코너(1,1), (1,4), (4,1), (4,4)도 승리 루트가 많음
    #       중앙 다음으로 중요
    # 3) 속성 분산
    #       한 줄(가로, 세로, 대각선)에 동일한 속성이 2개 이상 모이지 않게 초반에는 분산
    #       너무 빨리 한 속성 3개를 모으면, 상대가 그 속성의 마지막 말을 피해서 주거나, 막아버릴 수 있음

class P2():
    turn_count = 0
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.attributes = np.array(self.pieces)  # 속성을 NumPy 배열로 캐싱
        self.board = board
        self.available_pieces = available_pieces

    def select_piece(self):
        """
        상대방에게 줄 말을 선택합니다.
        - 바로 다음 턴에 내가 질 수 있는 말을 주지 않습니다.
        - 초반(턴 2 이하)에는 한 줄에 동일 속성이 2개 이상 몰릴 수 있는 말도 피합니다.
        - 그 외에는 미니맥스(알파-베타)로 상대가 가장 불리한 말을 줍니다.
        """
            # 첫 번째 말 선택은 랜덤
        if P2.turn_count == 0: 
            return random.choice(self.available_pieces)

        depth = self.adjust_depth()

        # 1. 즉시 패배를 유발하는 말은 제외
        safe_pieces = []
        for piece in self.available_pieces:
            if not self.opponent_can_win_next_turn(piece):
                safe_pieces.append(piece)

        # 2. 속성 분산: 초반(턴 2 이하)에는 한 줄에 동일 속성 2개 이상 몰릴 수 있는 말도 제외
        if P2.turn_count <= 2 and safe_pieces:
            filtered = []
            for piece in safe_pieces:
                risky = False
                for row in range(4):
                    for col in range(4):
                        if self.board[row][col] == 0:
                            temp_board = self.board.copy()
                            temp_board[row][col] = self.pieces.index(piece) + 1
                            if self.count_same_property_in_line(temp_board, piece, row, col) >= 2:
                                risky = True
                                break
                    if risky:
                        break
                if not risky:
                    filtered.append(piece)
            if filtered:
                safe_pieces = filtered

        # 3. 선택 후보가 남아있으면, 그중에서 미니맥스(알파-베타)로 상대가 가장 불리한 말을 줌
        candidate_pieces = safe_pieces if safe_pieces else self.available_pieces

        best_piece = None
        min_score = float('inf')
        for piece in candidate_pieces:
            # 미니맥스를 사용하여 상대방의 점수를 계산 (상대방이 두는 상황 가정 -> is_maximizing=False)
            score = self.minimax_select(piece, depth=depth, alpha=float('-inf'), beta=float('inf'), is_maximizing=False)
            # 최소 점수를 유발하는 말을 선택
            if score < min_score:
                min_score = score
                best_piece = piece

        # 4. 그래도 없으면 랜덤
        if best_piece is None:
            best_piece = random.choice(candidate_pieces)

        return best_piece

    def place_piece(self, selected_piece):
        """
        주어진 말을 보드에 배치합니다.
        모든 위치에서 미니맥스를 실행하되, 우선순위 위치에는 가중치를 부여합니다.
        """
        P2.turn_count += 1
        piece_index = self.pieces.index(selected_piece) + 1

        best_move = None
        max_score = float('-inf')
        depth = self.adjust_depth()

        for row, col in self.available_locations():
            self.board[row][col] = piece_index
            
            # 기본 미니맥스 점수
            base_score = self.minimax_place(self.board, depth=depth, alpha=float('-inf'), beta=float('inf'), is_maximizing=False)
            
            # 위치별 추가 보너스 (우선순위 위치에 더 높은 점수)
            position_bonus = 0
            if (row, col) in [(1,1), (1,2), (2,1), (2,2)]:  # 중앙
                position_bonus = 200
            elif (row, col) in [(0,0), (0,3), (3,0), (3,3)]:  # 코너
                position_bonus = 100
            elif (row, col) in [(0,1), (0,2), (1,0), (1,3), (2,0), (2,3), (3,1), (3,2)]:  # 가장자리 중앙
                position_bonus = 50
            
            final_score = base_score + position_bonus
            
            self.board[row][col] = 0

            if final_score > max_score:
                max_score = final_score
                best_move = (row, col)

        return best_move

# ===============================================================
#          상대가 바로 다음 턴에 이길 수 있는지 확인
# ===============================================================

    def opponent_can_win_next_turn(self, piece):
        piece_index = self.pieces.index(piece) + 1
        # 상대가 놓는 상황을 가정
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == 0:
                    self.board[row][col] = piece_index
                    if self.check_win(self.board):
                        self.board[row][col] = 0
                        return True
                    self.board[row][col] = 0
        return False

# ===============================================================
#          현재 턴에 바로 승리할 수 있는 수 확인
# ===============================================================

    def find_immediate_winning_move(self, piece_index):
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == 0:
                    self.board[row][col] = piece_index
                    if self.check_win(self.board):
                        self.board[row][col] = 0
                        return (row, col)
                    self.board[row][col] = 0
        return None

# ===============================================================
#          탐색 깊이 조정 함수
# ===============================================================

    def adjust_depth(self):        
        if P2.turn_count < 3: # 초반
            return 5
        elif P2.turn_count < 5: # 중반
            return 6
        else:                     # 후반
            return 7

# ===============================================================
#          미니맥스(알파-베타 가지치기) 알고리즘
# ===============================================================

    def minimax_select(self, piece, depth, alpha, beta, is_maximizing):
        """
        미니맥스(알파-베타 가지치기) 알고리즘 (select_piece용).
        piece: 현재 고려 중인 말
        depth: 탐색 깊이 제한
        alpha, beta: 알파-베타 가지치기 파라미터
        is_maximizing: 최대화 플레이어 여부
        """
        if depth == 0 or self.check_win(self.board):
            return self.evaluate(self.board)

        if is_maximizing:
            max_eval = float('-inf')
            piece_index = self.pieces.index(piece) + 1
            for row in range(4):
                for col in range(4):
                    if self.board[row][col] == 0:
                        self.board[row][col] = piece_index
                        eval_score = self.minimax_select(piece, depth - 1, alpha, beta, False)
                        self.board[row][col] = 0
                        max_eval = max(max_eval, eval_score)
                        alpha = max(alpha, eval_score)
                        if beta <= alpha:
                            break
            return max_eval
        else:
            # 상대방 턴 가정
            min_eval = float('inf')
            for row in range(4):
                for col in range(4):
                    if self.board[row][col] == 0:
                        self.board[row][col] = -1  # 상대방 가상의 말 배치
                        eval_score = self.minimax_select(piece, depth - 1, alpha, beta, True)
                        self.board[row][col] = 0
                        min_eval = min(min_eval, eval_score)
                        beta = min(beta, eval_score)
                        if beta <= alpha:
                            break
            return min_eval

# ===============================================================
#         미니맥스(알파-베타 가지치기) 알고리즘 (place_piece용)
# ===============================================================

    def minimax_place(self, board, depth, alpha, beta, is_maximizing):
        """
        미니맥스(알파-베타 가지치기) 알고리즘 (place_piece용).
        board: 현재 보드 상태
        depth: 탐색 깊이 제한
        alpha, beta: 알파-베타 파라미터
        is_maximizing: 최대화 플레이어 여부
        """
        if depth == 0 or self.check_win(board):
            return self.evaluate(board)

        if is_maximizing:
            max_eval = float('-inf')
            for row in range(4):
                for col in range(4):
                    if board[row][col] == 0:
                        board[row][col] = -1  # 플레이어의 가상 배치
                        eval_score = self.minimax_place(board, depth - 1, alpha, beta, False)
                        board[row][col] = 0
                        max_eval = max(max_eval, eval_score)
                        alpha = max(alpha, eval_score)
                        if beta <= alpha:
                            break
            return max_eval
        else:
            min_eval = float('inf')
            for row in range(4):
                for col in range(4):
                    if board[row][col] == 0:
                        board[row][col] = -1  # 상대방 가상의 말 배치
                        eval_score = self.minimax_place(board, depth - 1, alpha, beta, True)
                        board[row][col] = 0
                        min_eval = min(min_eval, eval_score)
                        beta = min(beta, eval_score)
                        if beta <= alpha:
                            break
            return min_eval

# ===============================================================
#         보드 상태 평가 함수
# ===============================================================

    def evaluate(self, board):
        """
        보드 상태를 평가하여 점수를 반환합니다.
        - 즉시승/즉시패, 한 수 뒤 승리, Fork, 위치 가중치, 라인 속성 일치 등 반영
        """
        # 1. 즉시승/즉시패 확인 (최우선)
        if self.check_win(board):
            return 100000000  # 즉시 승리
        
        # 상대가 다음에 이길 수 있는지 확인 (즉시 패배 위험)
        for piece in self.available_pieces:
            if self.opponent_can_win_next_turn(piece):
                return -100000000  # 즉시 패배 위험
        
        score = 0

        # # 2. 한 수 뒤 승리 가능성 평가
        # my_next_wins = self.count_next_turn_winning_opportunities(board)
        # opp_next_wins = self.count_opponent_next_turn_winning_opportunities(board)
        
        # score += my_next_wins * 1000      # 내가 한 수 뒤에 이길 기회
        # score -= opp_next_wins * 1000     # 상대가 한 수 뒤에 이길 기회

        # 3. Fork 점수
        my_fork = self.count_forks(board)
        opp_fork = self.count_forks(board, is_opponent=True)
        score += 500 * my_fork
        score -= 500 * opp_fork

        # 4. 위치별 가중치 (우선순위 위치에 보너스)
        score += self.calculate_position_bonus(board)

        # 5. 라인 점수 (속성 일치)
        for i in range(4):
            row = [board[i][j] for j in range(4)]
            col = [board[j][i] for j in range(4)]
            score += self.line_score(row)
            score += self.line_score(col)
        diag1 = [board[i][i] for i in range(4)]
        diag2 = [board[i][3 - i] for i in range(4)]
        score += self.line_score(diag1)
        score += self.line_score(diag2)

        return score

    def calculate_position_bonus(self, board):
        """
        위치별 가중치를 계산합니다.
        우선순위 위치에 말이 있으면 보너스 점수를 줍니다.
        """
        bonus = 0
        
        # 중앙 4칸 (가장 높은 가중치)
        center_positions = [(1,1), (1,2), (2,1), (2,2)]
        for (r, c) in center_positions:
            if board[r][c] > 0:
                bonus += 100  # 중앙 보너스
        
        # 코너 4칸 (중간 가중치)
        corner_positions = [(0,0), (0,3), (3,0), (3,3)]
        for (r, c) in corner_positions:
            if board[r][c] > 0:
                bonus += 50   # 코너 보너스
        
        # 가장자리 중앙 (낮은 가중치)
        edge_center_positions = [(0,1), (0,2), (1,0), (1,3), (2,0), (2,3), (3,1), (3,2)]
        for (r, c) in edge_center_positions:
            if board[r][c] > 0:
                bonus += 20   # 가장자리 중앙 보너스
        
        return bonus

# ===============================================================
#          라인/2x2 점수 계산 함수
# ===============================================================

    def line_score(self, line):
        """
        한 줄(가로, 세로, 대각선)의 속성 일치 정도에 따라 점수 부여
        """
        indices = [idx-1 for idx in line if idx > 0]
        if len(indices) < 2:
            return 0
        attributes = self.attributes[indices]
        score = 0
        for i in range(4):  # 각 속성별
            values = attributes[:, i]
            if len(set(values)) == 1:
                if len(indices) == 4:
                    score += 100
                elif len(indices) == 3:
                    score += 20
                elif len(indices) == 2:
                    score += 5
        return score

    def square_score(self, subgrid):
        if len(subgrid) < 4:
            return 0
        attributes = self.attributes[[idx-1 for idx in subgrid]]
        score = 0
        for i in range(4):
            if len(set(attributes[:, i])) == 1:
                score += 50
        return score

# ===============================================================
#           승리 조건 확인 함수
# ===============================================================

    def check_win(self, board=None):
        if board is None:
            board = self.board

        # 가로/세로 승리
        for i in range(4):
            row = [board[i][j] for j in range(4) if board[i][j] != 0]
            col = [board[j][i] for j in range(4) if board[j][i] != 0]
            if self.is_winning_line(row) or self.is_winning_line(col):
                return True

        # 대각선 승리
        diag1 = [board[i][i] for i in range(4) if board[i][i] != 0]
        diag2 = [board[i][3 - i] for i in range(4) if board[i][3 - i] != 0]
        if self.is_winning_line(diag1) or self.is_winning_line(diag2):
            return True

        # 2x2 사각형 승리
        for r in range(3):
            for c in range(3):
                subgrid = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
                subgrid = [idx for idx in subgrid if idx != 0]
                if self.is_winning_square(subgrid):
                    return True

        return False

    def is_winning_line(self, line):
        if len(line) < 4:
            return False
        attributes = self.attributes[[idx - 1 for idx in line]]
        for i in range(4):
            if len(set(attributes[:, i])) == 1:
                return True
        return False

    def is_winning_square(self, subgrid):
        if len(subgrid) < 4:
            return False
        attributes = self.attributes[[idx - 1 for idx in subgrid]]
        for i in range(4):
            if len(set(attributes[:, i])) == 1:
                return True
        return False

# ===============================================================
#          동일 속성 개수 세기
# ===============================================================

    def count_same_property_in_line(self, board, piece, row, col):
        piece_idx = self.pieces.index(piece) + 1
        board = board.copy()
        board[row][col] = piece_idx
        max_count = 0
        for lines in [
            [board[row, c] for c in range(4)],
            [board[r, col] for r in range(4)],
            [board[i, i] for i in range(4)] if row == col else [],
            [board[i, 3-i] for i in range(4)] if row + col == 3 else []
        ]:
            if not lines: continue
            indices = [idx-1 for idx in lines if idx > 0]
            if not indices: continue
            for k in range(4):
                values = [self.pieces[idx][k] for idx in indices]
                if len(values) > 1 and len(set(values)) == 1:
                    max_count = max(max_count, len(values))
        return max_count

# ===============================================================
#         보드 내 사용 가능한 위치 반환
# ===============================================================

    def available_locations(self):
        return [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0]
    
# ===============================================================
#           fork 개수 세기
# ===============================================================

    def count_forks(self, board, is_opponent=False):
        """
        Fork(두 개 이상의 승리 기회) 개수 세기
        """
        fork_count = 0
        # 빈칸마다 말을 놓아보고, 승리하는 경우의 수를 센다
        for row in range(4):
            for col in range(4):
                if board[row][col] == 0:
                    # 내 말/상대 말 중에서 남은 말 아무거나 하나를 가정해서 놓아봄
                    for piece in self.available_pieces:
                        idx = self.pieces.index(piece) + 1
                        board[row][col] = idx
                        if self.check_win(board):
                            fork_count += 1
                        board[row][col] = 0
                    # 한 칸에서 2개 이상 승리가 나오면 fork
                    if fork_count >= 2:
                        return 1
        return 0