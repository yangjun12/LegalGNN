# -*- coding: UTF-8 -*-

import torch
import scipy.sparse as sp
import numpy as np
from utils import utils
import torch.nn.functional as F
from torch.autograd import Variable
from models.BaseModel import BaseModel
from models.GNN_KG import GNN_KG
import torchsnooper
from sklearn import preprocessing
from models.BPR import BPR
import os
from collections import defaultdict
import random
import json
import copy

class LegalGNN(GNN_KG):
    reader = 'LegalGNNReader'
    runner = 'JointRunner'
    @staticmethod
    def parse_model_args(parser):
        # parser.add_argument('--layers', type=str, default='[64]',
        #                     help="Size of each layer.")
        # parser.add_argument('--dense_feature', type=int, default=0,
        #                     help="Use dense feature: 0: no use, N: dim.")
        parser.add_argument('--fix_size', type=int, default=16,
                            help="fix size length.")
        parser.add_argument('--trans', type=int, default=1,
                            help='0: transE 1: transR.')
        parser.add_argument('--transfer', type=int, default=2,
                            help='0: no trans 1: transfer.')
        parser.add_argument('--alpha', type=float, default=1,
                            help='rec_loss + alpha * trans_loss.')
        # parser.add_argument('--margin', type=float, default=0,
        #                     help='Margin in hinge loss.')
        # parser.add_argument('--feature_dims', type=int, default=64,
        #                     help="Feature dim in the first layer, int value.")
        # parser.add_argument('--sample_vector', type=str, default='[32,32]') #不支持两层数量不同
        # parser.add_argument('--node_dropout', type=str, default='[0.5,0.5]')
        return GNN_KG.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.fix_size = args.fix_size
        self.trans = args.trans
        self.margin = args.margin
        self.transfer = args.transfer # 1: only transfer rec, 2: transfer trans and rec
        self.alpha = args.alpha

        print("self.trans", self.trans)
        print("self.margin", self.margin)
        print("self.transfer", self.transfer)
        print("self.alpha", self.alpha)

        # self.margin = args.margin
        # self.layers = eval(args.layers)
        # self.feature_dims = corpus.n_entity
        # # self.feature_dims = corpus.n_items + corpus.n_users
        # self.n_entity = corpus.n_entity
        # self.n_relations = corpus.n_relations
        # self.sample_vector = eval(args.sample_vector)
        # self.node_dropout = eval(args.node_dropout)
        # self.dense_feature = args.dense_feature
        # self.emb_size = args.emb_size
        #
        # self.split_vector = [1]
        # self.split_mask = [6]
        # for n in self.sample_vector:
        #     self.split_vector.append(self.split_vector[-1] * n)
        # for n in self.sample_vector[1:]:
        #     self.split_mask.append(self.split_mask[-1] * n)
        # print("split_vector: ", self.split_vector)  # [1, 32, 1024]
        # print("split_mask: ", self.split_mask)      # [6, 192]
        # self.n_users = corpus.n_users
        # self.n_items = corpus.n_items

        # 0: normal sparse
        # 1: normal sparse + dense_1
        super().__init__(args, corpus)

    @torchsnooper.snoop()
    def _define_params(self):
        if self.fix_size < self.emb_size:
            self.n_embeddings = torch.nn.Embedding(self.n_entity, self.emb_size - self.fix_size)
        if self.fix_size > 0:
            # self.f_embeddings = torch.nn.Embedding(self.n_entity, self.fix_size)
            # self.f_embeddings.from_pretrained(torch.from_numpy(np.load('../data/LAW/entity_feature_' + str(self.fix_size) + '.npy')), freeze=True)
            np_fix = np.load('../data/LAW/entity_feature_' + str(self.fix_size) + '.npy')
            # np_fix = np.load('../data/LAW/entity_feature_768.npy')
            # np_fix_norm = preprocessing.normalize(np_fix, axis=0)

            if self.transfer != 0:
                np_fix_norm = (np_fix - np_fix.mean(0)) / np_fix.std(0)
            else:
                np_fix_norm = np_fix

            np_var = np.var(np_fix_norm, axis=0)
            print("np_var: ", np_var)
            self.f_embeddings = torch.nn.Embedding.from_pretrained(torch.from_numpy(np_fix_norm.astype(np.float32)), freeze=True)
            if self.transfer == 3:
                self.transfer_m = torch.nn.Linear(self.emb_size, self.emb_size)
            elif self.transfer in [1,2]:
                self.transfer_m = torch.nn.Linear(self.fix_size, self.fix_size)
                # self.transfer_m = torch.nn.Linear(768, self.fix_size)

        self.att = torch.nn.ModuleList([])
        for i in range(len(self.sample_vector)):
            L = torch.nn.Linear(self.emb_size * 2, self.emb_size)
            self.att.append(L)

        self.h_attention = torch.nn.Embedding(28, self.emb_size)

        self.m_att_w = torch.nn.ModuleList([])
        self.m_att_h = torch.nn.ModuleList([])
        for i in range(len(self.sample_vector)):
            L = torch.nn.Linear(self.emb_size, self.emb_size)
            self.m_att_w.append(L)

            V = torch.nn.Linear(self.emb_size, 1, bias=False)
            self.m_att_h.append(V)


        self.mlp = torch.nn.ModuleList([])
        for i in range(len(self.sample_vector)):
            L = torch.nn.Linear(self.emb_size * 2, self.emb_size, bias=True)
            self.mlp.append(L)
        self.layer_norm = torch.nn.LayerNorm(self.emb_size)

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bias = torch.nn.Embedding(self.n_entity, 1)

        # kg
        self.r_embeddings = torch.nn.Embedding(self.n_relations, self.emb_size)
        self.w_embeddings = torch.nn.Embedding(self.n_relations, self.emb_size * self.emb_size)
        # ↑ relation embedding: 0-buy, 1-complement, 2-substitute
        self.loss_function = torch.nn.MarginRankingLoss(margin=self.margin)

        # bpr
        self.u_embeddings = torch.nn.Embedding(self.n_users, self.emb_size)
        self.i_embeddings = torch.nn.Embedding(self.n_items, self.emb_size)
        self.user_bias = torch.nn.Embedding(self.n_users, 1)
        self.item_bias = torch.nn.Embedding(self.n_items, 1)
    # @torchsnooper.snoop()
    def forward(self, feed_dict):
        cat_neigh = feed_dict["neigh"].view(feed_dict["batch_size_rec"], 1 + feed_dict['neg_item'], -1).long()
        cat_mask = feed_dict["mask"].view(feed_dict["batch_size_rec"], 1 + feed_dict["neg_item"], -1, self.sample_vector[-1]).float()
        cat_r = feed_dict["r"].view(feed_dict["batch_size_rec"], 1 + feed_dict["neg_item"], -1).long()

        # print("cat_neigh: ", cat_neigh.shape, cat_neigh)
        # print("cat_mask: ", cat_mask.shape, cat_mask)
        # print("cat_r: ", cat_r.shape, cat_r)


        cat_neigh_list = torch.split(cat_neigh, self.split_vector, dim=-1)
        cat_mask_list = torch.split(cat_mask, self.split_mask, dim=-2)
        cat_r_list = torch.split(cat_r, self.split_mask, dim=-1)

        if self.fix_size == 0:
            cat_neigh_list = [self.n_embeddings(t) for t in cat_neigh_list]
        elif self.fix_size == self.emb_size:
            if self.transfer in [1,2]:
                cat_neigh_list = [self.transfer_m(self.f_embeddings(t)) for t in cat_neigh_list]
            else:
                cat_neigh_list = [self.f_embeddings(t) for t in cat_neigh_list]
        else:
            if self.transfer  in [1,2]:
                cat_neigh_list = [torch.cat((self.n_embeddings(t), self.transfer_m(self.f_embeddings(t))), dim=-1) for t in cat_neigh_list]
            else:
                cat_neigh_list = [torch.cat((self.n_embeddings(t), self.f_embeddings(t)), dim=-1) for t in cat_neigh_list]
        if self.transfer == 3:
            cat_neigh_list = [self.transfer_m(t) for t in cat_neigh_list]
        # cat_r_w_list = [self.w_attention(t) for t in cat_r_list]
        # cat_r_b_list = [self.b_attention(t) for t in cat_r_list]
        cat_r_h_list = [self.h_attention(t) for t in cat_r_list]
        # random_flag = np.random.rand() < 1
        random_flag = 0


        hidden = cat_neigh_list
        for layer in range(len(self.sample_vector), 0, -1): # layer = 2, 1
            next_hidden = []
            for hop in range(layer): # hop = 0, layer - 1
                neigh_vecs = hidden[hop + 1].reshape(
                    feed_dict["batch_size_rec"],
                    -1,
                    self.split_vector[hop],
                    self.sample_vector[hop],
                    self.emb_size) # 256, N, 32, emb_size | 16, 101, 8, 64 | 16, 101, 64, 64
                self_vecs = hidden[hop].reshape(
                    feed_dict["batch_size_rec"],
                    -1,
                    self.split_vector[hop],
                    1,
                    self.emb_size) # 256, N, emb_size | 16, 101, 1, 64 | 16, 101, 8, 64
                mask = cat_mask_list[hop].reshape(
                    feed_dict["batch_size_rec"],
                    -1,
                    self.split_vector[hop],
                    6,
                    self.sample_vector[hop]) # 256, N, 6, 32 | 16, 101, 6, 1 | 16, 101, 48, 8
                # r_w = cat_r_w_list[hop].reshape(
                #     feed_dict["batch_size"],
                #     -1,
                #     self.split_vector[hop],
                #     6,
                #     self.emb_size * 2,
                #     self.emb_size
                # ) # 16, 101, 6, 128
                # r_b = cat_r_b_list[hop].reshape(
                #     feed_dict["batch_size"],
                #     -1,
                #     self.split_vector[hop],
                #     6,
                #     self.emb_size
                # ) # 16, 101, 6
                r_h = cat_r_h_list[hop].reshape(
                    feed_dict["batch_size_rec"],
                    -1,
                    self.split_vector[hop],
                    6,
                    self.emb_size
                ) # 16, 101, 6
                if random_flag:
                    print("neigh_vecs: ", neigh_vecs.shape, neigh_vecs)
                    print("mask: ", mask.shape, mask)
                    print("self_vecs: ", self_vecs.shape, self_vecs)

                neigh_vecs = torch.einsum('ijpkv,ijpqk->ijpqv', neigh_vecs, mask) # 256, N, 6, emb_size | 16, 101, 6, 64
                mask_bool = torch.round(mask.sum(dim=-1)) # 256, N, 6 | 16, 101, 6
                if random_flag:
                    print("layer: ", layer, "mask_bool: ", mask_bool.shape, mask_bool)
                # attention: self -> r_neigh
                # print("mask_bool: ", mask_bool.shape, mask_bool) # torch.Size([8, 101, 1, 6])

                self_vecs_repeat = self_vecs.repeat(1,1,1,6,1) # 16, 101, 6, 64
                self_concat_neigh = torch.cat((self_vecs_repeat, neigh_vecs), dim=-1) # 16, 101, 6, 128
                # print("self_concat_neigh: ", self_concat_neigh.shape, self_concat_neigh) # torch.Size([8, 101, 1, 6, 128])

                attention_weight = self.att[layer - 1](self_concat_neigh) * r_h # # 256, N, 6
                # print("attention_weight: ", attention_weight.shape, attention_weight) # torch.Size([8, 101, 1, 6, 64])

                attention_weight = self.leaky_relu(attention_weight.sum(-1))
                attention_weight = attention_weight.masked_fill((1 - mask_bool).bool(), -1e32)  # 对mask位置为0的，填充-1e32, log之后变为了0


                if random_flag:
                    print("layer: ", layer, "attention_weight: ", attention_weight.shape, attention_weight)
                attention_weight = torch.nn.functional.softmax(attention_weight, dim=-1).masked_fill((1 - mask_bool).bool(), 0) # 对mask全部为0的，weight全部置为0
                if random_flag:
                    print("layer: ", layer, "attention_weight: ", attention_weight.shape, attention_weight)
                # attention_neigh_vector = (attention_weight.unsqueeze(-1) * neigh_vecs).sum(-2)   # 16, 101, 6, 64
                attention_neigh_vector = (attention_weight.unsqueeze(-1) * neigh_vecs).sum(-2)   # 16, 101, 6, 64

                # cat_emb = torch.cat((attention_neigh_vector, self_vecs.squeeze(-2)), dim=-1)
                #
                # alpha = self.m_att_h[layer - 1](torch.tanh(self.m_att_w[layer - 1](self_vecs.squeeze(-2))))
                # beta = self.m_att_h[layer - 1](torch.tanh(self.m_att_w[layer - 1](attention_neigh_vector)))
                #
                # alpha = torch.exp(alpha)
                # beta = torch.exp(beta)
                #
                # sum = alpha + beta + torch.ones_like(alpha) * 1e-13
                # alpha_t = alpha / sum
                # beta_t = beta / sum
                # print("alpha: ", alpha.shape)
                # print("self: ", self_vecs.shape)
                # print("attention_neigh_vector: ", attention_neigh_vector.shape)
                # attention_neigh_vector = alpha_t * self_vecs.squeeze(-2) + beta_t * attention_neigh_vector
                # print("attention_neigh_vector shape: ", attention_neigh_vector.shape)
                # print("self_vecs_repeat shape: ", self_vecs.squeeze(-2).shape)

                # attention_neigh_vector = (attention_neigh_vector + self_vecs.squeeze(-2)) / 2

                if random_flag:
                    print("layer: ", layer, "attention_neigh_vector: ", attention_neigh_vector.shape, attention_neigh_vector)
                # concat_embedding = torch.cat((self_vecs.squeeze(-2), attention_neigh_vector), dim=-1)
                # self_embedding = self.mlp[layer - 1](cat_emb)
                if random_flag:
                    print("layer: ", layer, "w_b", self.mlp[layer - 1].weight, self.mlp[layer - 1].bias)
                # self_embedding = self.leaky_relu(self_embedding)
                self_embedding = self.dropout(attention_neigh_vector)
                self_embedding = self.layer_norm(self_embedding)
                if random_flag:
                    print("layer: ", layer, "norm self_embedding: ", self_embedding.shape, self_embedding)
                next_hidden.append(self_embedding)

            hidden = next_hidden
        # print("hidden[0]", hidden[0].shape)
        final_emb = hidden[0].reshape(
            feed_dict["batch_size_rec"],
            -1,
            self.emb_size
        )
        # # print("final_emb: ", final_emb.shape, final_emb)
        user_emb, item_emb = torch.split(final_emb, [1, feed_dict["neg_item"]], dim=1)
        prediction_rec = (user_emb * item_emb).sum(dim=-1)
        if random_flag:
            print("prediction: ", prediction_rec.shape, prediction_rec)

        # cf_u_vectors = self.u_embeddings(feed_dict['user_id'])
        # cf_i_vectors = self.i_embeddings(feed_dict['item_id'])
        # u_bias = self.user_bias(feed_dict['user_id'])
        # i_bias = self.item_bias(feed_dict['item_id']).squeeze(-1)
        # prediction = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(dim=-1)  # [batch_size, -1]

        # u_bias = self.bias(feed_dict['user_id'])
        # i_bias = self.bias(feed_dict['item_id']).squeeze(-1)
        # prediction = prediction + u_bias + i_bias
        # return prediction_rec.view(feed_dict['batch_size_rec'], -1)
        if feed_dict['phase'] != "train":
            return prediction_rec.view(feed_dict['batch_size_rec'], -1)
        else:
            head_ids = feed_dict['head_id']  # [batch_size, -1]
            tail_ids = feed_dict['tail_id']  # [batch_size, -1]
            relation_ids = feed_dict['relation_id']  # [batch_size, -1]

            if self.fix_size == 0:
                head_vectors = self.n_embeddings(head_ids)
                tail_vectors = self.n_embeddings(tail_ids)
            elif self.fix_size == self.emb_size:
                if self.transfer == 2:
                    head_vectors = self.transfer_m(self.f_embeddings(head_ids))
                    tail_vectors = self.transfer_m(self.f_embeddings(tail_ids))
                else:
                    head_vectors = self.f_embeddings(head_ids)
                    tail_vectors = self.f_embeddings(tail_ids)
            else:
                if self.transfer == 2:
                    head_vectors = torch.cat((self.n_embeddings(head_ids), self.transfer_m(self.f_embeddings(head_ids))), dim=-1)
                    tail_vectors = torch.cat((self.n_embeddings(tail_ids), self.transfer_m(self.f_embeddings(tail_ids))), dim=-1)
                else:
                    head_vectors = torch.cat((self.n_embeddings(head_ids), self.f_embeddings(head_ids)), dim=-1)
                    tail_vectors = torch.cat((self.n_embeddings(tail_ids), self.f_embeddings(tail_ids)), dim=-1)

            if self.transfer == 3:
                head_vectors = self.transfer_m(head_vectors)
                tail_vectors = self.transfer_m(tail_vectors)

            relation_vectors = self.r_embeddings(relation_ids)
            if self.trans == 1:
                trans_w = self.w_embeddings(relation_ids).reshape(-1, 4, self.emb_size, self.emb_size)
                # print("head_vectors: ", head_vectors.shape, head_vectors)
                # print("trans_w: ", trans_w.shape, trans_w)
                head_vectors = torch.einsum('abc,abcd->abd', head_vectors, trans_w)
                tail_vectors = torch.einsum('abc,abcd->abd', tail_vectors, trans_w)
            # prediction_kg = -((head_vectors + relation_vectors - tail_vectors) ** 2).sum(-1)
            # prediction_kg = -(head_vectors + relation_vectors - tail_vectors).sum(-1)
            prediction_kg = self._calc(head_vectors, relation_vectors, tail_vectors)
            return prediction_kg.view(feed_dict['batch_size'], -1), prediction_rec.view(feed_dict['batch_size_rec'], -1)

    def _calc(self, h, r, t, p_norm=1): # p_norm = 1 for trans
        h = F.normalize(h, 2, -1)
        r = F.normalize(r, 2, -1)
        t = F.normalize(t, 2, -1)
        score = h + r - t
        return torch.norm(score, p_norm, -1)


    def loss(self, prediction_tuple):
        prediction_kg, prediction_rec = prediction_tuple
        # prediction_rec = prediction_tuple
        # loss_rec
        pos_pred, neg_pred = prediction_rec[:, 0], prediction_rec[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        neg_pred = (neg_pred * neg_softmax).sum(dim=1)
        loss = F.softplus(-(pos_pred - neg_pred)).mean()
        # loss_kg
        batch_size = prediction_kg.shape[0]
        pos_pred, neg_pred = prediction_kg[:, :2].flatten(), prediction_kg[:, 2:].flatten()
        if self.margin >= 0:
            loss += self.alpha * self.loss_function(pos_pred, neg_pred, utils.numpy_to_torch(self.margin * np.ones(batch_size * 2)))
        else:
            loss += self.alpha * F.softplus(-(pos_pred - neg_pred)).mean()
        return loss
