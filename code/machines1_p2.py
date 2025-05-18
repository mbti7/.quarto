import numpy as np
import torch
from itertools import product
from train.dqnagent import DQNAgent
from train.quartoenv import QuartoEnv

class P2():
    def __init__(self, board, available_pieces, selected_piece=None):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2)
                       for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board.copy()  # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces.copy()  # Currently available pieces
        self.selected_piece = selected_piece

        self.state_size = 34  # 보드 상태(16) + 남은 말(16) + 선택된 말(1) + 단계(1)
        self.agent = DQNAgent(state_size=self.state_size)
        self.agent.load_state_dict(torch.load("dqn_model.pth", map_location=torch.device('cpu'), weights_only=True))
        self.agent.eval()

    def select_piece(self):
        # 환경 상태 구성
        env = QuartoEnv()
        env.board = self.board.copy()
        env.available_pieces = self.available_pieces.copy()
        env.selected_piece = self.selected_piece
        env.phase = 'select_piece'

        state = env.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        with torch.no_grad():
            q_values = self.agent(state_tensor, 'select_piece')
            valid_actions = len(env.available_pieces)
            q_values = q_values[:valid_actions]
            action = torch.argmax(q_values).item()

        selected_piece = env.available_pieces[action]
        return selected_piece

    def place_piece(self, selected_piece):
        # 환경 상태 구성
        self.selected_piece = selected_piece
        env = QuartoEnv()
        env.board = self.board.copy()
        env.available_pieces = self.available_pieces.copy()
        env.selected_piece = selected_piece
        env.phase = 'place_piece'

        state = env.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        empty_positions = [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]

        with torch.no_grad():
            q_values = self.agent(state_tensor, 'place_piece')
            valid_actions = len(empty_positions)
            q_values = q_values[:valid_actions]
            action = torch.argmax(q_values).item()

        row, col = empty_positions[action]
        return (row, col)