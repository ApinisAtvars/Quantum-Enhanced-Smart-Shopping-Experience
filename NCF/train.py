import pandas as pd
import numpy as np
from NeuMF import NeuMFEngine
from data import SampleGenerator
from utils import set_global_seed

# Temp for automatically pushing when done training
import subprocess
import os

RATING_TYPE = 'implicit'  # 'explicit' or 'implicit'

neumf_config = {'alias': '07_04_subset_classical_baseline_nq2',
                'seed': 42,
                'num_epoch': 10, # original 100, less now because an epoch takes 40 min for implicit NeuMF
                'batch_size': 256, # original 256
                'optimizer': 'adam', # original 'adam'
                'adam_lr': 1e-3, # original 0.001
                'num_users': None,  # to be set after loading data
                'num_items': None,  # to be set after loading data
                'latent_dim_mf': 8, # original 8
                'latent_dim_mlp': 8, # original 8
                'num_negative': 4, # original 4
                'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector, this is what they used in the og paper too
                'l2_regularization': 0.0000001, # original 0
                'weight_init_gaussian': True,
                'use_cuda': True,
                'use_bachify_eval': True,
                'device_id': 0,
                'pretrain': False,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
                'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model',
                'data_path': r"data\ncf_preprocessed\ratings_subset.csv",
                'are_ratings_explicit': RATING_TYPE == 'explicit',
                'add_pre_post_net_layers': True, # Whether to add post_net, as in QVC/Dressed_Quantum_Net.py
                'n_qubits': 2, # Number of neurons in pre_net. Needed to have a baseline when n_qubits changed in QVC/train.py
                }

set_global_seed(neumf_config['seed'], use_cuda=neumf_config['use_cuda'])

# Load Data
def preprocess_data(dir: str, is_ml1m: bool) -> pd.DataFrame:
    """
    Preprocess the data, and set the number of users and items in the NeuMF config based on dataset size.

    Args:
        dir: The path to the ml32m dataset
    """
    if is_ml1m:
        dataset = pd.read_csv(dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    else:
        dataset = pd.read_csv(dir).rename(mapper={'userId': 'uid', 'itemId': 'mid'}, axis=1)
    user_id = dataset[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    dataset = pd.merge(dataset, user_id, on=['uid'], how='left')
    item_id = dataset[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    dataset = pd.merge(dataset, item_id, on=['mid'], how='left')
    dataset = dataset[['userId', 'itemId', 'rating', 'timestamp']]
    print('Range of userId is [{}, {}]'.format(dataset.userId.min(), dataset.userId.max()))
    print('Range of itemId is [{}, {}]'.format(dataset.itemId.min(), dataset.itemId.max()))

    neumf_config['num_users'] = dataset.userId.max() + 1
    neumf_config['num_items'] = dataset.itemId.max() + 1

    return dataset


dataset = preprocess_data(neumf_config['data_path'], False)

print(f"Number of users: {neumf_config['num_users']}, Number of items: {neumf_config['num_items']}")

# DataLoader for training

sample_generator = SampleGenerator(ratings=dataset, rating_type=RATING_TYPE)
evaluate_data = sample_generator.evaluate_data

train_size = len(sample_generator.train_ratings)
test_size = len(sample_generator.test_ratings)

config = neumf_config

engine = NeuMFEngine(config)

engine.log_data_split(train_size=train_size, test_size=test_size)

for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    if config['are_ratings_explicit']:
        mse = engine.evaluate(evaluate_data, epoch_id=epoch)
        engine.save(config['alias'], epoch, mse=mse)
    else:
        hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
        engine.save(config['alias'], epoch, hit_ratio=hit_ratio, ndcg=ndcg)
