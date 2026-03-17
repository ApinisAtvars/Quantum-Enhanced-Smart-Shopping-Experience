import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from typing import Literal, List

random.seed(0)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings, rating_type: Literal['explicit', 'implicit'] = 'implicit'):
        """
        Args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
            rating_type: Literal['explicit', 'implicit'], type of ratings
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.rating_type = rating_type
        self.ratings = ratings
        # explicit feedback using _normalize and implicit using _binarize
        # self.preprocess_ratings = self._normalize(ratings)
        self.preprocess_ratings = self._binarize(ratings) if self.rating_type == 'implicit' else self._normalize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        # create negative item samples for NCF learning
        self.negatives = self._sample_negative(ratings) if self.rating_type == 'implicit' else None # Wasted time on negative sampling for explicit feedback
        self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings
    
    def _binarize(self, ratings):
        """binarize into 0 or 1, implicit feedback"""
        ratings = deepcopy(ratings)
        ratings.loc[ratings['rating'] > 0, 'rating'] = 1.0 # replace ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings

    def _split_loo(self, ratings):
        """leave one out train/test split """
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    # Original implementation, consumes too much RAM when ml32m is used
    # def _sample_negative(self, ratings):
    #     """return all negative items & 100 sampled negative items"""
    #     interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
    #         columns={'itemId': 'interacted_items'})
    #     interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
    #     interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(list(x), 99))
    #     return interact_status[['userId', 'negative_items', 'negative_samples']]

    def _sample_negative(self, ratings):
        """return interacted items & 99 sampled negative items (memory efficient)"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        item_pool_list = list(self.item_pool)
        interact_status['negative_samples'] = interact_status['interacted_items'].apply(
            lambda x: self._rejection_sample(item_pool_list, x, 99))
        return interact_status[['userId', 'interacted_items', 'negative_samples']]

    def _rejection_sample(self, pool_list, exclusion_set, n):
        """Sample n items from pool_list not in exclusion_set via rejection sampling"""
        samples = []
        while len(samples) < n:
            candidate = random.choice(pool_list)
            if candidate not in exclusion_set:
                samples.append(candidate)
        return samples

    def instance_a_train_loader(self, num_negatives, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'interacted_items']], on='userId') if self.rating_type == 'implicit' else self.train_ratings
        item_pool_list = list(self.item_pool)
        if self.rating_type == 'implicit':
            train_ratings['negatives'] = train_ratings['interacted_items'].apply(
                lambda x: self._rejection_sample(item_pool_list, x, num_negatives))
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            if self.rating_type == 'implicit':
                for i in range(num_negatives):
                    users.append(int(row.userId))
                    items.append(int(row.negatives[i]))
                    ratings.append(float(0))  # negative samples get 0 rating
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self) -> List[torch.LongTensor]:
        """Create evaluation data

        Returns:
            A list of tensors containing evaluation data.\n
            If rating_type is 'implicit', returns [test_users, test_items, negative_users, negative_items].\n
            If rating_type is 'explicit', returns [test_users, test_items, test_ratings].
        """
        if self.rating_type == 'implicit':
            test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
            test_users, test_items, negative_users, negative_items = [], [], [], []
            for row in test_ratings.itertuples():
                test_users.append(int(row.userId))
                test_items.append(int(row.itemId))
                for i in range(len(row.negative_samples)):
                    negative_users.append(int(row.userId))
                    negative_items.append(int(row.negative_samples[i]))
            return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                    torch.LongTensor(negative_items)]
        else:
            test_users = torch.LongTensor(self.test_ratings['userId'].tolist())
            test_items = torch.LongTensor(self.test_ratings['itemId'].tolist())
            test_ratings = torch.FloatTensor(self.test_ratings['rating'].tolist())
            return [test_users, test_items, test_ratings]
