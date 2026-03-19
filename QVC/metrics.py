import math
import pandas as pd


class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        self.set_subjects(subjects)

    def _validate_lengths(self, subject_columns):
        expected_length = len(subject_columns[0])
        if any(len(column) != expected_length for column in subject_columns[1:]):
            raise ValueError('All subject columns must have the same length.')

    def _build_explicit_subjects(self, subjects):
        if len(subjects) != 4:
            raise ValueError('Explicit evaluation expects [test_users, test_items, test_ratings, pred_scores].')

        test_users, test_items, test_ratings, pred_scores = subjects
        self._validate_lengths([test_users, test_items, test_ratings, pred_scores])

        full = pd.DataFrame({
            'user': test_users,
            'item': test_items,
            'test_score': test_ratings,
            'score': pred_scores,
        })
        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)
        return full

    def _build_implicit_subjects(self, subjects):
        if len(subjects) != 6:
            raise ValueError(
                'Implicit evaluation expects [test_users, test_items, test_scores, negative_users, negative_items, negative_scores].'
            )

        test_users, test_items, test_scores, neg_users, neg_items, neg_scores = subjects
        self._validate_lengths([test_users, test_items, test_scores])
        self._validate_lengths([neg_users, neg_items, neg_scores])

        test = pd.DataFrame({
            'user': test_users,
            'test_item': test_items,
            'test_score': test_scores,
        })
        full = pd.DataFrame({
            'user': neg_users + test_users,
            'item': neg_items + test_items,
            'score': neg_scores + test_scores,
        })
        full = pd.merge(full, test, on=['user'], how='left')
        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)
        return full

    def set_subjects(self, subjects, is_explicit=None):
        """
        Args:
            subjects: list containing evaluation data.
                Implicit: [test_users, test_items, test_scores, negative_users, negative_items, negative_scores]
                Explicit: [test_users, test_items, test_ratings, pred_scores]
            is_explicit: optional bool. If omitted, infer from the subject list length.
        """
        if not isinstance(subjects, list):
            raise TypeError('subjects must be provided as a list.')

        if is_explicit is None:
            if len(subjects) == 4:
                is_explicit = True
            elif len(subjects) == 6:
                is_explicit = False
            else:
                raise ValueError('Unable to infer evaluation mode from subjects.')

        self._subjects = self._build_explicit_subjects(subjects) if is_explicit else self._build_implicit_subjects(subjects)

    def cal_hit_ratio(self):
        """Hit Ratio @ top_K"""
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        return len(test_in_top_k) * 1.0 / full['user'].nunique()

    def cal_ndcg(self):
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']].copy()
        test_in_top_k.loc[:, 'ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()

    def cal_mse(self):
        full = self._subjects
        return ((full['score'] - full['test_score']) ** 2).mean()