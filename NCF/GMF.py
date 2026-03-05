# General Matrix Factorization module
# source: https://github.com/hexiangnan/neural_collaborative_filtering/blob/master/GMF.py

import numpy as np
import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, latent_dim: int):
        """
        :param num_users: Number of users in the dataset
        :param num_items: Number of items in the dataset
        :param latent_dim: Size of embedding vector for both user and item embeddings
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        
        self.user_embedding = nn.Embedding(
            num_embeddings= self.num_users,
            embedding_dim= self.latent_dim 
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=self.latent_dim
        )
        self.pred_layer = nn.Linear(in_features=self.latent_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, user_indices, item_indices):

        user_vector = self.user_embedding(user_indices)
        item_vector = self.item_embedding(item_indices)

        predict_vector = torch.mul(user_vector, item_vector)
        prediction = self.pred_layer(predict_vector)

        # return self.sigmoid(prediction)
        return prediction.squeeze() # Because the dataset uses explicit feedback

    # def get_model(self, num_users:int, num_items:int, latent_dim:int, reg_params:list = None):
    #     """
    #     :param num_users: Number of users in the dataset
    #     :param num_items: Number of items in the dataset
    #     :param latent_dim: Size of embedding vector for both user and item embeddings
    #     :param reg_params: (Unused) List of 2 ints - 1st L2 reg for user embeddings, 2nd L2 reg for item embeddings
    #     """
    #     usr_embedding = nn.Embedding(
    #         num_embeddings=num_users,
    #         embedding_dim=latent_dim,
    #     )
    #     itm_embedding = nn.Embedding(
    #         num_embeddings=num_items,
    #         embedding_dim=latent_dim
    #     )

    #     predict_vector = torch.mul(usr_embedding, itm_embedding)

    #     model = nn.Sequential(
    #         predict_vector,
    #         nn.Linear(predict_vector.shape, 1),
    #         nn.Sigmoid()
    #     )

    #     return model

