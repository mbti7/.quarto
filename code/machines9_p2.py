# machines_p2.py
import numpy as np
import random
from itertools import product

class P2:
    turn_count = 0

    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l)
                       for i in range(2)
                       for j in range(2)
                       for k in range(2)
                       for l in range(2)]
        self.attributes = np.array(self.pieces)
        self.board = [list(row) for row in board]
        self.available_pieces = available_pieces

    def select_piece(self):
        safe_pieces = [p for p in self.available_pieces if not self.opponent_can_win_next_turn(p)]
        candidates = safe_pieces if safe_pieces else self.available_pieces

        # 속성 위험도 줄이기
        if P2.turn_count <= 2:
            filtered = []
            for p in candidates:
                if not self.creates_risk(p):
                    filtered.append(p)
            if filtered:
                candidates = filtered

        best, best_score = None, float('inf')
        depth = self.adjust_depth()
        for p in candidates:
            score = self.minimax_select(p, depth, -float('inf'), float('inf'), False)
            if score < best_score:
                best_score, best = score, p
        return best

    def opponent_can_win_next_turn(self, piece):
        piece_index = self.pieces.index(piece) + 1
        for r, c in product(range(4), range(4)):
            if self.board[r][c] == 0:
                self.board[r][c] = piece_index
                if self.check_win():
                    self.board[r][c] = 0
                    return True
                self.board[r][c] = 0
        return False
    
    def adjust_depth(self):
        if P2.turn_count < 3:
            return 5
        elif P2.turn_count < 5:
            return 6
        else:
            return 7


    def place_piece(self, selected_piece):
        P2.turn_count += 1
        piece_index = self.pieces.index(selected_piece) + 1
        locs = [(r, c) for r, c in product(range(4), range(4)) if self.board[r][c] == 0]

        # 1) 즉시 승리
        for r, c in locs:
            self.board[r][c] = piece_index
            if self.check_win():
                self.board[r][c] = 0
                return (r, c)
            self.board[r][c] = 0

        # 2) 상대 즉시 승리 차단
        for r, c in locs:
            self.board[r][c] = piece_index
            threat = any(self.simulate_win(opp) for opp in self.available_pieces)
            self.board[r][c] = 0
            if not threat:
                return (r, c)

        # 3) Minimax로 전략 수 싸움
        best_move, best_score = None, float('-inf')
        depth = self.adjust_depth()
        for r, c in locs:
            self.board[r][c] = piece_index
            score = self.minimax_place(depth, -float('inf'), float('inf'), False)
            self.board[r][c] = 0
            if score > best_score:
                best_score, best_move = score, (r, c)
        return best_move

    def simulate_win(self, piece):
        idx = self.pieces.index(piece) + 1
        for r, c in product(range(4), range(4)):
            if self.board[r][c] == 0:
                self.board[r][c] = idx
                if self.check_win():
                    self.board[r][c] = 0
                    return True
                self.board[r][c] = 0
        return False

    def creates_risk(self, piece):
        idx = self.pieces.index(piece) + 1
        for r, c in product(range(4), range(4)):
            if self.board[r][c] == 0:
                self.board[r][c] = idx
                if self.count_same_property(r, c) >= 2:
                    self.board[r][c] = 0
                    return True
                self.board[r][c] = 0
        return False

    def count_same_property(self, r, c):
        max_count = 0
        lines = [
            [self.board[r][i] for i in range(4)],
            [self.board[i][c] for i in range(4)],
        ]
        if r == c:
            lines.append([self.board[i][i] for i in range(4)])
        if r + c == 3:
            lines.append([self.board[i][3 - i] for i in range(4)])

        for line in lines:
            pieces = [self.pieces[i-1] for i in line if i > 0]
            for k in range(4):
                attr = [p[k] for p in pieces]
                if len(attr) > 1 and len(set(attr)) == 1:
                    max_count = max(max_count, len(attr))
        return max_count

    def minimax_select(self, piece, depth, alpha, beta, is_max):
        if depth == 0 or self.check_win():
            return self.evaluate()

        if is_max:
            max_eval = float('-inf')
            idx = self.pieces.index(piece) + 1
            for r, c in product(range(4), range(4)):
                if self.board[r][c] == 0:
                    self.board[r][c] = idx
                    score = self.minimax_select(piece, depth-1, alpha, beta, False)
                    self.board[r][c] = 0
                    max_eval = max(max_eval, score)
                    alpha = max(alpha, score)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float('inf')
            for r, c in product(range(4), range(4)):
                if self.board[r][c] == 0:
                    self.board[r][c] = -1
                    score = self.minimax_select(piece, depth-1, alpha, beta, True)
                    self.board[r][c] = 0
                    min_eval = min(min_eval, score)
                    beta = min(beta, score)
                    if beta <= alpha:
                        break
            return min_eval

    def minimax_place(self, depth, alpha, beta, is_max):
        if depth == 0 or self.check_win():
            return self.evaluate()

        if is_max:
            max_eval = float('-inf')
            for r, c in product(range(4), range(4)):
                if self.board[r][c] == 0:
                    self.board[r][c] = -1
                    score = self.minimax_place(depth-1, alpha, beta, False)
                    self.board[r][c] = 0
                    max_eval = max(max_eval, score)
                    alpha = max(alpha, score)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float('inf')
            for r, c in product(range(4), range(4)):
                if self.board[r][c] == 0:
                    self.board[r][c] = -1
                    score = self.minimax_place(depth-1, alpha, beta, True)
                    self.board[r][c] = 0
                    min_eval = min(min_eval, score)
                    beta = min(beta, score)
                    if beta <= alpha:
                        break
            return min_eval

    def evaluate(self):
        score = 0
        for i in range(4):
            row = [self.board[i][j] for j in range(4) if self.board[i][j] > 0]
            col = [self.board[j][i] for j in range(4) if self.board[j][i] > 0]
            score += self.line_score(row) + self.line_score(col)

        diag1 = [self.board[i][i] for i in range(4) if self.board[i][i] > 0]
        diag2 = [self.board[i][3-i] for i in range(4) if self.board[i][3-i] > 0]
        score += self.line_score(diag1) + self.line_score(diag2)

        return score

    def line_score(self, line):
        if len(line) < 2:
            return 0
        pieces = self.attributes[[i-1 for i in line]]
        score = 0
        for i in range(4):
            s = set(pieces[:, i])
            if len(s) == 1:
                score += 20
            elif len(s) == 2:
                score += 5
        return score

    def check_win(self):
        for i in range(4):
            row = [self.board[i][j] for j in range(4)]
            col = [self.board[j][i] for j in range(4)]
            if self.is_winning(row) or self.is_winning(col):
                return True
        diag1 = [self.board[i][i] for i in range(4)]
        diag2 = [self.board[i][3-i] for i in range(4)]
        return self.is_winning(diag1) or self.is_winning(diag2)

    def is_winning(self, line):
        if line.count(0) > 0 or len(line) < 4:
            return False
        attrs = self.attributes[[i-1 for i in line]]
        for i in range(4):
            if len(set(attrs[:, i])) == 1:
                return True
        return False
