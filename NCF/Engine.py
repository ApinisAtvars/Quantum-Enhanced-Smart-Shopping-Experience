#source: https://github.com/yihong-chen/neural-collaborative-filtering/blob/master/src/engine.py
import torch
import tqdm

class Engine:
    def __init__(self, learning_rate, l2_regularization):
        self.crit = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=l2_regularization
        )

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model')
        # Assume code is run on CUDA-capable hardware
        users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss
    
    def train_epoch()
