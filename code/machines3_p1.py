import numpy as np

class P1():
    turn_count = 0
    def __init__(self, board, available_pieces):
        """
        P1 클래스 초기화
        board: 현재 게임 판 상태 (4x4 numpy array)
        available_pieces: 현재 선택 가능한 말들의 리스트
        """
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.attributes = np.array(self.pieces)  # 속성을 NumPy 배열로 캐싱
        self.board = board
        self.available_pieces = available_pieces

    def select_piece(self):
        """
        상대방에게 줄 말을 선택합니다.
        - 바로 다음 턴에 내가 질 수 있는 말을 주지 않습니다.
        - 미니맥스 + 알파-베타 가지치기를 사용합니다.
        """
        depth = self.adjust_depth()


        # 먼저, 상대방이 바로 다음 턴에 이 말로 승리 가능한지 체크하여 그런 말은 제외
        safe_pieces = []
        for piece in self.available_pieces:
            if not self.opponent_can_win_next_turn(piece):
                safe_pieces.append(piece)

        # 모든 말이 다 위험하다면 어쩔 수 없이 하나 선택 (이 경우 최소점 유발 말)
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

        return best_piece

    def place_piece(self, selected_piece):
        """
        주어진 말을 보드에 배치합니다.
        - 바로 놓아서 이길 수 있다면 그 수를 즉시 둡니다.
        - 그렇지 않다면 미니맥스 + 알파-베타 가지치기를 사용합니다.
        """
        
        P1.turn_count += 1
        
        piece_index = self.pieces.index(selected_piece) + 1

        # 1. 즉시 승리 가능한 수 확인
        winning_move = self.find_immediate_winning_move(piece_index)
        if winning_move is not None:
            
            return winning_move

        # 2. 즉시 승리가 불가능하면 미니맥스 탐색
        best_move = None
        max_score = float('-inf')
        depth = self.adjust_depth()


        for row in range(4):
            for col in range(4):
                if self.board[row][col] == 0:
                    # 가상의 배치
                    self.board[row][col] = piece_index
                    # 상대의 반응을 가정 (is_maximizing=False)
                    score = self.minimax_place(self.board, depth=depth, alpha=float('-inf'), beta=float('inf'), is_maximizing=False)
                    # 되돌리기
                    self.board[row][col] = 0

                    # 최대 점수를 유발하는 위치 선택
                    if score > max_score:
                        max_score = score
                        best_move = (row, col)

        
        return best_move

    def opponent_can_win_next_turn(self, piece):
        """
        해당 piece를 상대에게 줬을 때, 상대가 바로 다음 턴에 승리할 수 있는지 확인합니다.
        상대방의 턴을 가정하고, 이 piece를 보드의 가능한 모든 위치에 둬봅니다.
        만약 어떤 위치에 두었을 때 바로 승리한다면 True를 반환합니다.
        """
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

    def find_immediate_winning_move(self, piece_index):
        """
        현재 턴에 piece_index를 놓아서 바로 승리할 수 있는 수가 있는지 확인합니다.
        승리 가능한 위치를 찾으면 그 위치를 반환하고, 없으면 None을 반환합니다.
        """
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == 0:
                    self.board[row][col] = piece_index
                    if self.check_win(self.board):
                        self.board[row][col] = 0
                        return (row, col)
                    self.board[row][col] = 0
        return None

    def adjust_depth(self):
        """
        턴 수와 게임 상태에 따라 탐색 깊이를 동적으로 조정
        """
        
        if P1.turn_count < 3: # 초반
            return 5
        elif P1.turn_count < 5: # 중반
            return 6
        else:                     # 후반
            return 7


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

    def evaluate(self, board):
        """
        보드 상태를 평가하여 점수를 반환합니다.
        동일 특성을 가지는 라인/2x2는 점수를 높게 책정.
        """
        score = 0

        # 가로, 세로 라인 평가
        for i in range(4):
            row = [board[i][j] for j in range(4) if board[i][j] != 0]
            col = [board[j][i] for j in range(4) if board[j][i] != 0]
            score += self.line_score(row)
            score += self.line_score(col)

        # 대각선 평가
        diag1 = [board[i][i] for i in range(4) if board[i][i] != 0]
        diag2 = [board[i][3 - i] for i in range(4) if board[i][3 - i] != 0]
        score += self.line_score(diag1)
        score += self.line_score(diag2)

        # 2x2 사각형 평가
        for r in range(3):
            for c in range(3):
                subgrid = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
                subgrid = [idx for idx in subgrid if idx != 0]
                score += self.square_score(subgrid)

        return score

    def line_score(self, line):
        """
        한 줄에서 동일한 속성이 얼마나 많은지 기반으로 점수를 계산합니다.
        """
        if len(line) < 2:
            return 0
        attributes = self.attributes[[idx-1 for idx in line]]
        score = 0
        for i in range(4):
            # 모든 말이 동일 속성이면 가중치 증가
            if len(set(attributes[:, i])) == 1:
                score += len(line) * 10
        return score

    def square_score(self, subgrid):
        """
        2x2 사각형에서 동일한 속성이 얼마나 많은지 기반으로 점수를 계산합니다.
        """
        if len(subgrid) < 4:
            return 0
        attributes = self.attributes[[idx-1 for idx in subgrid]]
        score = 0
        for i in range(4):
            if len(set(attributes[:, i])) == 1:
                score += 50
        return score

    def check_win(self, board=None):
        """
        현재 보드 상태에서 승리 조건을 확인합니다.
        """
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
        """
        한 줄(가로, 세로, 대각선)에서 모든 말의 특정 속성이 동일한 경우 승리.
        """
        if len(line) < 4:
            return False
        attributes = self.attributes[[idx - 1 for idx in line]]
        for i in range(4):
            if len(set(attributes[:, i])) == 1:
                return True
        return False

    def is_winning_square(self, subgrid):
        """
        2x2 사각형이 승리 조건을 충족하는지 확인합니다.
        """
        if len(subgrid) < 4:
            return False
        attributes = self.attributes[[idx - 1 for idx in subgrid]]
        for i in range(4):
            if len(set(attributes[:, i])) == 1:
                return True
        return False