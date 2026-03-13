import pandas as pd
import numpy as np
from NeuMF import NeuMFEngine
from data import SampleGenerator

# Temp for automatically pushing when done training
import subprocess
import os

RATING_TYPE = 'implicit'  # 'explicit' or 'implicit'

neumf_config = {'alias': 'neumf_initial_test_implicit',
                'num_epoch': 2, # original 100, less now because an epoch takes 40 min for implicit NeuMF
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
                'are_ratings_explicit': RATING_TYPE == 'explicit'
                }

# Load Data
# ml1m_dir = 'data/ml-1m/ratings.dat'
ml32m_dir = 'data/ncf_preprocessed/ratings.csv'
# ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
ml32m_rating = pd.read_csv(ml32m_dir).rename(mapper={'userId': 'uid', 'itemId': 'mid'}, axis=1)
# Reindex
user_id = ml32m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml32m_rating = pd.merge(ml32m_rating, user_id, on=['uid'], how='left')
item_id = ml32m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
ml32m_rating = pd.merge(ml32m_rating, item_id, on=['mid'], how='left')
ml32m_rating = ml32m_rating[['userId', 'itemId', 'rating', 'timestamp']]
print('Range of userId is [{}, {}]'.format(ml32m_rating.userId.min(), ml32m_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(ml32m_rating.itemId.min(), ml32m_rating.itemId.max()))

# Set num_users and num_items in config
neumf_config['num_users'] = ml32m_rating.userId.max() + 1
neumf_config['num_items'] = ml32m_rating.itemId.max() + 1

print(f"Number of users: {neumf_config['num_users']}, Number of items: {neumf_config['num_items']}")

# DataLoader for training

sample_generator = SampleGenerator(ratings=ml32m_rating, rating_type=RATING_TYPE)
evaluate_data = sample_generator.evaluate_data
# Specify the exact model
# config = gmf_config
# engine = GMFEngine(config)
# config = mlp_config
# engine = MLPEngine(config)
config = neumf_config
engine = NeuMFEngine(config)
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


print("Training complete! Initializing auto-sync to Git...")

# Path to the powershell script (assumes it's in the same directory as this python script)
ps_script_path = os.path.join(os.getcwd(), "git-sync.ps1")

try:
    # Run the PowerShell script
    # -ExecutionPolicy Bypass ensures the script runs even if your system restricted scripts
    result = subprocess.run(
        ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", ps_script_path, "-CommitMessage", "AUTOMATED: Done training implicit NeuMF"],
        capture_output=True,
        text=True,
        check=True
    )
    
    # Print the output from the PowerShell script so you can see the Git results
    print(result.stdout)

except subprocess.CalledProcessError as e:
    print(f"Error occurred while syncing to Git: {e.stderr}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")