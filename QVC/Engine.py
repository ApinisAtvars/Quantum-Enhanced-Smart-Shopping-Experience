import torch
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import save_checkpoint, use_optimizer
from metrics import MetronAtK
from tqdm import tqdm


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self.are_ratings_explicit = config['are_ratings_explicit']
        self._metron = MetronAtK(top_k=10)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        # self.crit = torch.nn.BCELoss()

        self.crit = torch.nn.MSELoss() if config['are_ratings_explicit'] else torch.nn.BCELoss()

    def log_data_split(self, train_size, test_size):
        """Log train/test split statistics once per run."""
        total_size = train_size + test_size
        if total_size == 0:
            return

        train_pct = (train_size / total_size) * 100.0
        test_pct = (test_size / total_size) * 100.0

        self._writer.add_scalar('data_split/train_percent', train_pct, 0)
        self._writer.add_scalar('data_split/test_percent', test_pct, 0)
        self._writer.add_scalar('data_split/train_size', train_size, 0)
        self._writer.add_scalar('data_split/test_size', test_size, 0)

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        for i, batch in enumerate(tqdm(train_loader, leave=False)):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            global_step = i + epoch_id * num_batches
            self._writer.add_scalar('model/individual_loss', loss, global_step)
            total_loss += loss
        self._writer.add_scalar('model/loss_per_epoch', total_loss, epoch_id)

    def _predict(self, users, items):
        if self.config['use_bachify_eval'] == False:
            return self.model(users, items)

        scores = []
        bs = self.config['batch_size']
        for start_idx in range(0, len(users), bs):
            end_idx = min(start_idx + bs, len(users))
            batch_users = users[start_idx:end_idx]
            batch_items = items[start_idx:end_idx]
            scores.append(self.model(batch_users, batch_items))
        return torch.concatenate(scores, dim=0)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            if self.are_ratings_explicit == False:
                negative_users, negative_items = evaluate_data[2], evaluate_data[3]
                test_ratings = None
            else:
                test_ratings = evaluate_data[2]
            if self.config['use_cuda'] is True:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                if self.are_ratings_explicit == False:
                    negative_users = negative_users.cuda()
                    negative_items = negative_items.cuda()
                if self.are_ratings_explicit:
                    test_ratings = test_ratings.cuda()

            test_scores = self._predict(test_users, test_items)
            if self.are_ratings_explicit == False:
                negative_scores = self._predict(negative_users, negative_items)

            if self.config['use_cuda'] is True:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                test_scores = test_scores.cpu()
                if self.are_ratings_explicit == False:
                    negative_users = negative_users.cpu()
                    negative_items = negative_items.cpu()
                    negative_scores = negative_scores.cpu()
                else:
                    test_ratings = test_ratings.cpu()

            if self.are_ratings_explicit == False:
                self._metron.set_subjects([
                    test_users.data.view(-1).tolist(),
                    test_items.data.view(-1).tolist(),
                    test_scores.data.view(-1).tolist(),
                    negative_users.data.view(-1).tolist(),
                    negative_items.data.view(-1).tolist(),
                    negative_scores.data.view(-1).tolist(),
                ], is_explicit=False)
            else:
                self._metron.set_subjects([
                    test_users.data.view(-1).tolist(),
                    test_items.data.view(-1).tolist(),
                    test_ratings.data.view(-1).tolist(),
                    test_scores.data.view(-1).tolist(),
                ], is_explicit=True)
        if self.are_ratings_explicit == False:
            hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
            self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
            self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
            print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))
            return hit_ratio, ndcg
        else:
            mse = self._metron.cal_mse()
            self._writer.add_scalar('performance/MSE', mse, epoch_id)
            print('[Evluating Epoch {}] MSE = {:.4f}'.format(epoch_id, mse))
            return mse

    def save(self, alias, epoch_id, hit_ratio=None, ndcg=None, mse=None):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.are_ratings_explicit:
            metric_value = mse if mse is not None else hit_ratio
            model_template = self.config.get('explicit_model_dir', 'checkpoints/{}_Epoch{}_MSE{:.4f}.model')
            model_dir = model_template.format(alias, epoch_id, metric_value)
        else:
            model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)