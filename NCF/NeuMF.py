import torch
from torch import nn

from GMF import GMF
from MLP import MLP


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim_gmf, latent_dim_mlp, layers = [64,32,16,8]):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim_gmf = latent_dim_gmf
        self.latent_dim_mlp = latent_dim_mlp

        self.user_embedding_gmf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_gmf)
        self.item_embedding_gmf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_gmf)

        self.user_embedding_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.item_embedding_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)

        self.fc_layers = nn.ModuleList()
        for in_size, out_size in zip(layers[:-1], layers[1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        self.pred_layer = nn.Linear(layers[-1] + self.latent_dim_gmf, 1)

        # Initialize the weights with Gaussian dist, as per https://github.com/yihong-chen/neural-collaborative-filtering/blob/master/src/neumf.py
        # Mean 0, STD 0.01
        for sm in self.modules():
            if isinstance(sm, (nn.Embedding, nn.Linear)):
                nn.init.normal_(sm.weight.data, 0.0, 0.01)

    
    def forward(self, user_indices, item_indices):
        # Same as in GMF.py
        user_embedding_gmf = self.user_embedding_gmf(user_indices)
        item_embedding_gmf = self.item_embedding_gmf(item_indices)

        # Same as in MLP.py
        user_embedding_mlp = self.user_embedding_mlp(user_indices)
        item_embedding_mlp = self.item_embedding_mlp(item_indices)

        # Same as in both module files
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mf_vector = torch.mul(user_embedding_gmf, item_embedding_gmf)

        # Same as in MLP.py
        for i, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[i](mlp_vector)
            mlp_vector = nn.ReLU()(mlp_vector)
        
        # New to this file, concatenating outputs to make one prediction
        final_vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        prediction = self.pred_layer(final_vector)

        return prediction.squeeze()
    




