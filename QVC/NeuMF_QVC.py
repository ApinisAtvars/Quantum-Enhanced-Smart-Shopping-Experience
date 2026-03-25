import torch
from Engine import Engine
from utils import use_cuda, resume_checkpoint
from Gaussian_Dressed_Quantum_Net import GaussianDressedQuantumNetwork

from torch import nn
import pennylane as qml



class QVCNeuMF(torch.nn.Module):
    def __init__(self, config):
        super(QVCNeuMF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']
        self.are_ratings_explicit = config['are_ratings_explicit']

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.final_mlp_ff_layer = torch.nn.Linear(in_features=config['layers'][-2], out_features=config['latent_dim_mlp'])

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.logistic = torch.nn.Sigmoid() # Only applied if the ratings are implicit

        self.quantum_network = GaussianDressedQuantumNetwork(config) # Assigned to affine_output after weights are loaded
        # Initialize model parameters with a Gaussian distribution (with a mean of 0 and standard deviation of 0.01)
        if config['weight_init_gaussian'] and not config['pretrain']:
            for sm in self.modules():
                if isinstance(sm, (nn.Embedding, nn.Linear)):
                    print(sm)
                    torch.nn.init.normal_(sm.weight.data, 0.0, 0.01)

    
    #region Forward
    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers)-1)):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)
        
        mlp_vector = self.final_mlp_ff_layer(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        if self.are_ratings_explicit:
            rating = logits.view(-1)
        else:
            rating = self.logistic(logits)
        return rating
    #endregion

class QVCNeuMFEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = QVCNeuMF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        if config['pretrain'] and config['pretrain_qvrn_dir'] is None: # Load weights here if the pretrained model is a vanilla NeuMF
            resume_checkpoint(self.model, model_dir=config['pretrain_neumf_dir'])
            
        # Freeze only the embedding layers.
        for param in self.model.parameters():
            if param in [self.model.embedding_user_mlp, self.model.embedding_item_mlp, self.model.embedding_user_mf, self.model.embedding_item_mf]:
                param.requires_grad = False


        # Insert the quantum circuit before the final affine output layer
        self.model.final_mlp_ff_layer = nn.Sequential(self.model.final_mlp_ff_layer, 
                                                      self.model.quantum_network)

        # ensure quantum network parameters are trainable
        for param in self.model.final_mlp_ff_layer.parameters():
            param.requires_grad = True
        
        if config['pretrain'] and config['pretrain_qvrn_dir'] is not None: # Load weights here if the pretrained model is a QVCNeuMF
            resume_checkpoint(self.model, model_dir=config['pretrain_qvrn_dir'])

        # initialize engine (optimizer, etc.) after setting up the model's frozen/trainable parameters
        super(QVCNeuMFEngine, self).__init__(config)
        print(self.model)


if __name__=="__main__":
    RATING_TYPE = 'implicit'  # 'explicit' or 'implicit'

    neumf_config = {'alias': 'test',
                    'num_epoch': 100, # original 100, less now because an epoch takes 40 min for implicit NeuMF
                    'batch_size': 256, # original 256
                    'optimizer': 'adam', # original 'adam'
                    'adam_lr': 1e-3, # original 0.001
                    'num_users': 1,  # doesn't matter
                    'num_items': 1,  # doesn't matter
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
    
    # NeuMFEngine(neumf_config)
    qml.StronglyEntanglingLayers