# Multi-Layer Perceptron module
# source: https://github.com/hexiangnan/neural_collaborative_filtering/blob/master/MLP.py

import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, num_users: int, num_items: int, latent_dim: int, layers = [64,32,16,8]):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.fc_layers = layers
        self.latent_dim = latent_dim

        self.user_embedding = nn.Embedding(
            num_embeddings=self.num_users,
            embedding_dim=self.latent_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=self.latent_dim
        )

        # For creating FC layers using for loop
        self.fc_layers = nn.ModuleList()

        for in_size, out_size in zip(layers[:-1], layers[1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        self.pred_layer = nn.Linear(layers[-1], 1)

    
    def forward(self, user_indices, item_indices):
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)

        vector = torch.cat([user_embedding, item_embedding], dim=-1)

        for i, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[i](vector)
            vector = nn.ReLU()(vector)
        
        prediction = self.pred_layer(vector)

        return prediction.squeeze()
        

