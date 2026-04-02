import pandas as pd
import numpy as np
from NeuMF_QVC import QVCNeuMFEngine
from data import SampleGenerator
from utils import set_global_seed

# Temp for automatically pushing when done training
import subprocess
import os

RATING_TYPE = 'implicit'  # 'explicit' or 'implicit'
'''
e - epoch
b - batch size
nq - num qubits
qd - qubit depth


'''
neumf_config = {'alias': '02_04_no_pretrain_AngleEmbedding_qd2_nq4',
                'seed': 42,
                'num_epoch': 10,                    # original 100
                'batch_size': 256,                    # original 256
                'optimizer': 'adam',                # original 'adam'
                'adam_lr': 1e-3,                    # original 0.001
                'clip_grad_norm_': None,            # set to None for no gradient clipping, original None
                # 'sgd_lr': 0.003,                    # new for sgd
                # 'sgd_momentum': 0,                  # disable momentum for finetuning
                'num_users': None,                  # to be set after loading data
                'num_items': None,                  # to be set after loading data
                'latent_dim_mf': 8,                 # original 8
                'latent_dim_mlp': 8,                # original 8
                'num_negative': 4,                  # original 4
                'layers': [16, 64, 32, 16, 8],      # layers[0] is the concat of latent user vector & latent item vector, this is what they used in the og paper too
                'l2_regularization': 0,     # original 0
                'weight_init_gaussian': True,
                'use_cuda': False,
                'use_bachify_eval': True,
                'device_id': 0,
                'pretrain': False,
                'pretrain_neumf_dir': None,     # does nothing if qvrn_dir provided
                'pretrain_qvrn_dir': None,    # if provided, will load weights for whole system including quantum circuit dressing
                'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model',
                'are_ratings_explicit': RATING_TYPE == 'explicit',
                'n_qubits': 4,
                'q_depth': 2,                       # Number of variational layers
                'q_delta': 0.01,                     # Initial spread of random quantum weights
                'data_path': r"data\ncf_preprocessed\ratings_subset.csv",
                'description': "Train AngleEmbedding circuit from scratch, this time with set seed."
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


dataset = preprocess_data(neumf_config['data_path'], is_ml1m=False)

print(f"Number of users: {neumf_config['num_users']}, Number of items: {neumf_config['num_items']}")

# DataLoader for training

sample_generator = SampleGenerator(ratings=dataset, rating_type=RATING_TYPE)
evaluate_data = sample_generator.evaluate_data

train_size = len(sample_generator.train_ratings)
test_size = len(sample_generator.test_ratings)

config = neumf_config

engine = QVCNeuMFEngine(config)

engine.log_data_split(train_size=train_size, test_size=test_size)

engine.log_model_architecture(sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size']))

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
