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
base_config = {
                'seed': 2505,
                'num_epoch': 10,                    # original 100
                'batch_size': 256,                    # original 256
                'optimizer': 'adam',                # original 'adam'
                'adam_lr': 1e-3,                    # original 0.001
                'clip_grad_norm_': None,            # set to None for no gradient clipping, original None
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
                'data_path': r"data\ncf_preprocessed\ratings_subset_500.csv"
                }


experiments = [
    # {
    #     'circuit_type': 'strongly_entangling',
    #     'use_residual': True,
    #     'n_qubits': 4,
    #     'q_depth': 2,
    #     'q_delta': 0.01,
    #     'alias': '500_res_difseed2_res_se_base',
    #     'description': 'Baseline winner: Strongly entangling (nq=4, qd=2) with residual connection.'
    # },
    # {
    #     'circuit_type': 'strongly_entangling',
    #     'use_residual': True,
    #     'n_qubits': 6,
    #     'q_depth': 2,
    #     'q_delta': 0.01,
    #     'alias': '500_res_difseed2_res_se_wider',
    #     'description': 'Wider circuit: Increased qubits to 6 for higher dimensional quantum latent space.'
    # },
    # {
    #     'circuit_type': 'mps',
    #     'use_residual': True,
    #     'n_qubits': 4,
    #     'q_depth': 4,
    #     'q_delta': 0.01,
    #     'alias': '500_res_difseed2_res_se_deeper',
    #     'description': 'Deeper circuit: Increased depth to 4 for stronger entanglement.'
    # },
    {
        'circuit_type': 'strongly_entangling',
        'use_residual': True,
        'n_qubits': 4,
        'q_depth': 2,
        'q_delta': 0.1,
        'alias': '500_res_difseed2_se_high_delta',
        'description': 'Higher initialization spread: q_delta=0.1 to start with a less uniform state. Trained on subset of 500 users'
    },
    {
        'circuit_type': 'data_reuploading',
        'use_residual': True,
        'n_qubits': 4,
        'q_depth': 2,
        'alias': '500_res_difseed2_data_reup',
        'description': 'Data re-uploading circuit WITH residual connection to test its native expressivity. Trained on subset of 500 users'
    },
    {
        'circuit_type': 'mps',
        'use_residual': True,
        'n_qubits': 4,
        'q_depth': 4,
        'alias': '500_res_difseed2_mps_deep',
        'description': 'Tree Tensor Network (MPS) WITH residual connection and deeper circuit. Trained on subset of 500 users'
    }
]

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

    base_config['num_users'] = dataset.userId.max() + 1
    base_config['num_items'] = dataset.itemId.max() + 1

    return dataset


dataset = preprocess_data(base_config['data_path'], is_ml1m=False)

print(f"Number of users: {base_config['num_users']}, Number of items: {base_config['num_items']}")

# DataLoader for training

sample_generator = SampleGenerator(ratings=dataset, rating_type=RATING_TYPE)
evaluate_data = sample_generator.evaluate_data

train_size = len(sample_generator.train_ratings)
test_size = len(sample_generator.test_ratings)

for i, exp_config in enumerate(experiments):
    print("=" * 80)
    print(f"Starting Experiment {i + 1}/{len(experiments)}: {exp_config['alias']}")
    print(f"Description: {exp_config['description']}")
    print("=" * 80)

    # Merge base config with experiment config
    config = base_config.copy()
    config.update(exp_config)

    set_global_seed(config['seed'], use_cuda=config['use_cuda'])

    engine = QVCNeuMFEngine(config)

    if i == 0:
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
