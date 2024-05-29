import torch

class Config:
    def __init__(self):
        self.batch_size = 128
        self.block_size = 4
        self.max_iters = 5000
        self.eval_interval = 200
        self.learning_rate = 3e-4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval_iters = 100
        self.n_embd = 768
        self.n_head = 6
        self.n_layer = 6
        self.dropout = 0.2