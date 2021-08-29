# -*- coding: UTF-8 -*-

import torch
import scipy.sparse as sp
import numpy as np
from utils import utils
import torch.nn.functional as F
from torch.autograd import Variable
from models.BaseModel import BaseModel
import torchsnooper
from models.BPR import BPR
from collections import defaultdict
import random
import json
import copy

class GNN_KG(BPR):
    reader = 'LegalGNNReader'
    runner = 'JointRunner'
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--layers', type=str, default='[64]',
                            help="Size of each layer.")
        parser.add_argument('--dense_feature', type=int, default=0,
                            help="Use dense feature: 0: no use, N: dim.")
        # parser.add_argument('--fix_size', type=int, default=0,
        #                     help="fix size length.")
        # parser.add_argument('--trans', type=float, default=0,
        #                     help='0: transE 1: transR.')
        parser.add_argument('--margin', type=float, default=1,
                            help='Margin in hinge loss.')
        parser.add_argument('--feature_dims', type=int, default=64,
                            help="Feature dim in the first layer, int value.")
        parser.add_argument('--sample_vector', type=str, default='[32,32]') #不支持两层数量不同
        parser.add_argument('--node_dropout', type=str, default='[0.5,0.5]')
        return BPR.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.margin = args.margin
        self.layers = eval(args.layers)
        self.feature_dims = corpus.n_entity
        # self.feature_dims = corpus.n_items + corpus.n_users
        self.n_entity = corpus.n_entity
        self.n_relations = corpus.n_relations
        self.sample_vector = eval(args.sample_vector)
        self.node_dropout = eval(args.node_dropout)
        self.dense_feature = args.dense_feature
        self.emb_size = args.emb_size
        self.split_vector = [1]
        self.split_mask = [6]
        for n in self.sample_vector:
            self.split_vector.append(self.split_vector[-1] * n)
        for n in self.sample_vector[1:]:
            self.split_mask.append(self.split_mask[-1] * n)
        print("split_vector: ", self.split_vector)  # [1, 32, 1024]
        print("split_mask: ", self.split_mask)      # [6, 192]
        self.n_users = corpus.n_users
        self.n_items = corpus.n_items

        # 0: normal sparse
        # 1: normal sparse + dense_1
        super().__init__(args, corpus)

    @torchsnooper.snoop()
    def _define_params(self):
        self.n_embeddings = torch.nn.Embedding(self.n_entity, self.emb_size)

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
            # torch.nn.init.normal_(L.weight, mean=0, std=0.01)
            # torch.nn.init.normal_(L.bias, mean=0, std=0.01)
            self.mlp.append(L)
        self.layer_norm = torch.nn.LayerNorm(self.emb_size)

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bias = torch.nn.Embedding(self.n_entity, 1)

        # kg
        self.r_embeddings = torch.nn.Embedding(self.n_relations, self.emb_size)
        # ↑ relation embedding: 0-buy, 1-complement, 2-substitute
        self.loss_function = torch.nn.MarginRankingLoss(margin=self.margin)

        # bpr
        self.u_embeddings = torch.nn.Embedding(self.n_users, self.emb_size)
        self.i_embeddings = torch.nn.Embedding(self.n_items, self.emb_size)
        self.user_bias = torch.nn.Embedding(self.n_users, 1)
        self.item_bias = torch.nn.Embedding(self.n_items, 1)
    @torchsnooper.snoop()
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

        cat_neigh_list = [self.n_embeddings(t) for t in cat_neigh_list]
        # cat_r_w_list = [self.w_attention(t) for t in cat_r_list]
        # cat_r_b_list = [self.b_attention(t) for t in cat_r_list]
        cat_r_h_list = [self.h_attention(t) for t in cat_r_list]
        # random_flag = np.random.rand() < 1
        random_flag = 0
        if random_flag:
            print("user_neigh: ", user_neigh.shape, user_neigh)
            print("item_neigh: ", item_neigh.shape, item_neigh)

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

            head_vectors = self.n_embeddings(head_ids)
            tail_vectors = self.n_embeddings(tail_ids)
            relation_vectors = self.r_embeddings(relation_ids)

            # prediction_kg = -((head_vectors + relation_vectors - tail_vectors) ** 2).sum(-1)
            prediction_kg = -(head_vectors + relation_vectors - tail_vectors).sum(-1)
            return prediction_kg.view(feed_dict['batch_size'], -1), prediction_rec.view(feed_dict['batch_size_rec'], -1)

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
        loss += self.loss_function(pos_pred, neg_pred, utils.numpy_to_torch(np.ones(batch_size * 2)))
        return loss

    class Dataset_kg(BaseModel.Dataset):
        def _prepare(self):
            if self.phase == 'train':
                self.data = utils.df_to_dict(self.corpus.relation_df)
                self.neg_heads = np.zeros(len(self), dtype=int)
                self.neg_tails = np.zeros(len(self), dtype=int)
            # print("self.data in dataset_kg: ", type(self.data), self.phase, self.data)
            super()._prepare()
            n_bound = [[0, 3327],   # related to data
                       [3327, 347193],
                       [347193, 375165],
                       [375165, 377379],
                       [377379, 378620],
                       [378620, 432117],
                       [432117, 436161]]
            self.index_a = [-1] * self.corpus.n_entity
            self.index_b = [-1] * self.corpus.n_entity
            for a, b in n_bound:
                for index in range(a, b):
                    self.index_a[index] = a
                    self.index_b[index] = b
            # self.sample = self.model.sample

        def _get_feed_dict(self, index):
            if self.phase == 'train':
                head, tail = self.data['head'][index], self.data['tail'][index]
                relation = self.data['relation'][index]
                head_id = np.array([head, head, head, self.neg_heads[index]])
                tail_id = np.array([tail, tail, self.neg_tails[index], tail])
                relation_id = np.array([relation] * 4)
            else:
                target_item = self.data['item_id'][index]
                neg_items = self.neg_items[index]
                tail_id = np.concatenate([[target_item], neg_items])
                head_id = self.data['user_id'][index] * np.ones_like(tail_id)

                relation_id = np.ones_like(tail_id) * 7  # related to data, relation
            feed_dict = {'head_id': head_id, 'tail_id': tail_id, 'relation_id': relation_id}
            return feed_dict

        def negative_sampling(self):

            sample_type = 4
            # 1: 全部entity进行sample
            # 3: user, item, 其它
            # 4: 对每一种类型的entity进行sample
            item_link_set = set([2, 3, 4, 8])

            # for i in range(10):
            #     print("{}\t{}\t{}".format(self.data['head'][i], self.data['relation'][i], self.data['tail'][i]))
            # exit()
            if sample_type == 1:
                for i in range(len(self)):
                    head, tail, relation = self.data['head'][i], self.data['tail'][i], self.data['relation'][i]
                    self.neg_tails[i] = np.random.randint(0, self.corpus.n_entity)
                    self.neg_heads[i] = np.random.randint(0, self.corpus.n_entity)
                    while (head, relation, self.neg_tails[i]) in self.corpus.triplet_set:
                        self.neg_tails[i] = np.random.randint(0, self.corpus.n_entity)
                    while (self.neg_heads[i], relation, tail) in self.corpus.triplet_set:
                        self.neg_heads[i] = np.random.randint(0, self.corpus.n_entity)
            elif sample_type == 2:
                for i in range(len(self)):
                    head, tail, relation = self.data['head'][i], self.data['tail'][i], self.data['relation'][i]
                    self.neg_tails[i] = np.random.randint(self.corpus.min_item, self.corpus.n_entity)
                    if relation == 7:  # 这个是临时的，因为原来处理数据的时候写成了7, 2-2个地方
                        self.neg_heads[i] = np.random.randint(0, self.corpus.n_users)
                        while (head, relation, self.neg_tails[i]) in self.corpus.triplet_set:
                            self.neg_tails[i] = np.random.randint(self.corpus.min_item, self.corpus.n_entity)
                        while (self.neg_heads[i], relation, tail) in self.corpus.triplet_set:
                            self.neg_heads[i] = np.random.randint(0, self.corpus.n_users)
                    else:
                        self.neg_heads[i] = np.random.randint(self.corpus.min_item, self.corpus.n_entity)
                        while (head, relation, self.neg_tails[i]) in self.corpus.triplet_set:
                            self.neg_tails[i] = np.random.randint(self.corpus.min_item, self.corpus.n_entity)
                        while (self.neg_heads[i], relation, tail) in self.corpus.triplet_set:
                            self.neg_heads[i] = np.random.randint(self.corpus.min_item, self.corpus.n_entity)
            elif sample_type == 3:
                for i in range(len(self)):
                    head, tail, relation = self.data['head'][i], self.data['tail'][i], self.data['relation'][i]
                    if relation == 7:  # 这个是临时的，因为原来处理数据的时候写成了7, 2-2个地方
                        self.neg_heads[i] = np.random.randint(0, self.corpus.n_users)
                        self.neg_tails[i] = np.random.randint(self.corpus.min_item, self.corpus.max_item)
                        while (head, relation, self.neg_tails[i]) in self.corpus.triplet_set:
                            self.neg_tails[i] = np.random.randint(self.corpus.min_item, self.corpus.max_item)
                        while (self.neg_heads[i], relation, tail) in self.corpus.triplet_set:
                            self.neg_heads[i] = np.random.randint(0, self.corpus.n_users)
                    elif relation in item_link_set:
                        self.neg_heads[i] = np.random.randint(self.corpus.min_item, self.corpus.max_item)
                        self.neg_tails[i] = np.random.randint(self.corpus.max_item, self.corpus.n_entity)
                        while (head, relation, self.neg_tails[i]) in self.corpus.triplet_set:
                            self.neg_tails[i] = np.random.randint(self.corpus.max_item, self.corpus.n_entity)
                        while (self.neg_heads[i], relation, tail) in self.corpus.triplet_set:
                            self.neg_heads[i] = np.random.randint(self.corpus.min_item, self.corpus.max_item)
                    else:
                        self.neg_heads[i] = np.random.randint(self.corpus.max_item, self.corpus.n_entity)
                        self.neg_tails[i] = np.random.randint(self.corpus.max_item, self.corpus.n_entity)
                        while (head, relation, self.neg_tails[i]) in self.corpus.triplet_set:
                            self.neg_tails[i] = np.random.randint(self.corpus.max_item, self.corpus.n_entity)
                        while (self.neg_heads[i], relation, tail) in self.corpus.triplet_set:
                            self.neg_heads[i] = np.random.randint(self.corpus.max_item, self.corpus.n_entity)
            elif sample_type == 4:
                for i in range(len(self)):
                    head, tail, relation = self.data['head'][i], self.data['tail'][i], self.data['relation'][i]
                    self.neg_heads[i] = np.random.randint(self.index_a[head], self.index_b[head])
                    self.neg_tails[i] = np.random.randint(self.index_a[tail], self.index_b[tail])
                    while (head, relation, self.neg_tails[i]) in self.corpus.triplet_set:
                        self.neg_tails[i] = np.random.randint(self.index_a[tail], self.index_b[tail])
                    while (self.neg_heads[i], relation, tail) in self.corpus.triplet_set:
                        self.neg_heads[i] = np.random.randint(self.index_a[head], self.index_b[head])

    class Dataset(BaseModel.Dataset):
        def _prepare(self):
            self.sample_vector = self.model.sample_vector
            print("self.sample_vector: ", self.sample_vector)
            self.node_r_niegh_array = self.corpus.neigh_r # s_node, r, n_node
            self.node_r_niegh = {}
            self.node_type_list = self.corpus.node_type_list # [0,...0,1,...,1,....]
            self.node_type_item = self.corpus.node_type_dic # {"user": 0}

            self.max_neigh_r = max([len(self.node_r_niegh_array[t]) for t in self.node_r_niegh_array])
            print("before max relation: ")
            self.n_relation = max([max(self.node_r_niegh_array[t].keys()) for t in self.node_r_niegh_array if self.node_r_niegh_array[t]]) + 1
            print("n_relation: ", self.n_relation)
            print("len of self.node_type_item: ", len(self.node_type_item))
            self.r_t_m = np.ones((len(self.node_type_item), self.n_relation), dtype=np.int) * -1# [node_type][r_o]=[r_t] # 6 * 28 -> max_neigh_r
            node_relation_type_map = defaultdict(set)

            for node in self.node_r_niegh_array:
                self.node_r_niegh[node] = {}
                for r in self.node_r_niegh_array[node]:
                    self.node_r_niegh_array[node][r] = np.array(self.node_r_niegh_array[node][r])
                    # print("node_r_niegh_array:  ", node, r, self.node_r_niegh_array[node][r].shape, self.node_r_niegh_array[node][r])
                    self.node_r_niegh[node][r] = set(self.node_r_niegh_array[node][r])
                    node_relation_type_map[self.node_type_list[node]].add(r)
            for node in node_relation_type_map:
                r_list = sorted(node_relation_type_map[node])
                for index, r in enumerate(r_list):
                    self.r_t_m[node][r] = index
            print(self.r_t_m)
            self.node_dropout = self.model.node_dropout  # [0.5, 0.5]: [user, query]
            self.split_dic = {}


        def split_integer(self, m, n):
            assert n > 0
            quotient = int(m / n)
            remainder = m % n
            if remainder > 0:
                return np.array([quotient] * (n - remainder) + [quotient + 1] * remainder)
            return np.array([quotient] * n)

        def get_neighbor(self, s_node):
            # print("type s_node: ", type(s_node), s_node)
            if type(s_node) == np.int64:
                o_vector = [np.asarray([[s_node]])] # shape: (1,1)
                # print("s_node: 1", s_node)
            else:
                o_vector = [np.expand_dims(np.asarray(s_node), -1)] # len(shape) must >= 2, for t()
            input_size = o_vector[0].shape[0]

            item_query_flag = random.random() < 0.5 # true: 保留pos_item-query, 去掉user-query.
            pp_user = o_vector[0][0][0]
            pp_item = o_vector[0][1][0]
            # print("pp_user: ", pp_user)
             #print("pp_item: ", pp_item)
            r_u_i = 7  # related to data
            r_i_u = 21
            r_q_i = 1
            r_i_q = 15
            r_u_q = 0
            r_q_u = 14
            u_query = set()
            i_query = set()
            if pp_user in self.node_r_niegh and r_u_q in self.node_r_niegh[pp_user]:
                u_query |= self.node_r_niegh[pp_user][r_u_q]
            if pp_item in self.node_r_niegh and r_i_q in self.node_r_niegh[pp_item]:
                i_query |= self.node_r_niegh[pp_item][r_i_q]
            common_query = u_query & i_query

            trap_set = common_query | set([pp_user, pp_item])
            o_mask = []
            o_r = []
            # print(self.__dict__)
            for n_sample in self.sample_vector:
                t_vector = []
                t_mask = []
                t_r = []
                # print("t input: ", o_vector[-1].shape)
                for s_index, s_n in enumerate(o_vector[-1].reshape(-1)):
                    # print("s_n: ", s_n)
                    n_vector = []
                    n_mask = [0] * self.max_neigh_r
                    n_r = [0] * self.max_neigh_r
                    n_all_r = set(self.node_r_niegh_array[s_n])
                    rand_float = [random.random(), random.random()]
                    if self.phase == "train":
                        if rand_float[0] < self.node_dropout[0]:
                            n_all_r.discard(21)
                        if rand_float[1] < self.node_dropout[1]:
                            n_all_r.discard(15)
                        ###### 先discard, 后判断
                        if s_n in trap_set:
                            if s_n == pp_user:
                                if r_u_i in n_all_r and not (self.node_r_niegh[s_n][r_u_i] - set([pp_item])):
                                    n_all_r.discard(r_u_i)
                                    # print("discard_7: ", pp_user, pp_item, self.node_r_niegh[s_n][r_u_i])
                            if s_n == pp_item:
                                if r_i_u in n_all_r and not (self.node_r_niegh[s_n][r_i_u] - set([pp_user])):
                                    n_all_r.discard(r_i_u)
                                    # print("discard_21: ", pp_user, pp_item, self.node_r_niegh[s_n][r_i_u])
                            if item_query_flag: # True: 保留item_query
                                if s_n in common_query:
                                    if r_q_u in n_all_r and not (self.node_r_niegh[s_n][r_q_u] - set([pp_user])):
                                        n_all_r.discard(r_q_u)
                                if s_n == pp_item:
                                    if r_u_q in n_all_r and not (self.node_r_niegh[s_n][r_u_q] - common_query):
                                        n_all_r.discard(r_u_q)
                            else:
                                if s_n in common_query:
                                    if r_q_i in n_all_r and not (set(self.node_r_niegh[s_n][r_q_i]) - set([pp_item])):
                                        n_all_r.discard(r_q_i)
                                if s_n == pp_user:
                                    if r_i_q in n_all_r and not (set(self.node_r_niegh[s_n][r_i_q]) - common_query):
                                        n_all_r.discard(r_i_q)

                        # 如果去掉user-pos_item的连边后，user-item没有连边，把这种关系去掉

                        # 如果去掉user-query的连边后，user-query没有连边，则把这种关系去掉

                        # 如果去掉pos_item-query连边后，pos_item-query没有连边，则把这种关系去掉

                    if not n_all_r:
                        t_vector.extend([0] * n_sample)
                        t_mask.extend(n_mask)
                        t_r.extend(n_r)
                        continue
                    # print("n_all_r", n_all_r)

                    # 确定每个关系下取多少, 查表
                    if (n_sample, len(n_all_r)) in self.split_dic:
                        neigh_per_r = self.split_dic[(n_sample, len(n_all_r))]
                    else:
                        neigh_per_r = self.split_integer(n_sample, len(n_all_r))
                        self.split_dic[(n_sample, len(n_all_r))] = neigh_per_r
                    # print("neigh_per_r: ", neigh_per_r)
                    for i, r in enumerate(n_all_r):
                        mapped_r = self.r_t_m[self.node_type_list[s_n]][r]
                        if mapped_r >= 0:
                            n_mask[mapped_r] = neigh_per_r[i]
                            n_r[mapped_r] = r
                    #     print("r,mapped_r: ", r, mapped_r)
                    # print("n_r: ", n_r)

                    for r_index, r_sample in enumerate(n_mask):
                        if r_sample: # 挑选selected
                            # 判断当前节点类型：
                            # 如果是p_user, 去掉候选集合中的p_item
                            # 如果是p_item, 去掉候选集合中的p_user
                            # 如果是p_user, 而且有query连边，根据flag看是否去
                            # 如果是p_item, 而且有query连边，根据flag看是否去
                            temp_r = n_r[r_index]

                            change_flag = 0
                            if self.phase == "train" and s_n in trap_set:
                                if s_n == pp_user and r_u_i == temp_r:
                                    candidate_list = self.node_r_niegh[s_n][r_u_i] - set([pp_item])
                                    change_flag = 1
                                if s_n == pp_item and r_i_u == temp_r:
                                    candidate_list = self.node_r_niegh[s_n][r_i_u] - set([pp_user])
                                    change_flag = 1
                                if item_query_flag:  # True: 保留item_query
                                    if s_n in common_query and r_q_u == temp_r:
                                        candidate_list = self.node_r_niegh[s_n][r_q_u] - set([pp_user])
                                        change_flag = 1
                                    if s_n == pp_item and r_i_u == temp_r:
                                        candidate_list = self.node_r_niegh[s_n][r_i_u] - common_query
                                        change_flag = 1
                                else:
                                    if s_n in common_query and r_q_i == temp_r:
                                        candidate_list = self.node_r_niegh[s_n][r_q_i] - set([pp_item])
                                        change_flag = 1
                                    if s_n == pp_user and r_i_q == temp_r:
                                        candidate_list = self.node_r_niegh[s_n][r_i_q] - common_query
                                        change_flag = 1
                            if change_flag:
                                candidate_list = np.array(list(candidate_list))
                            #     # print("candidate_list 1: ", type(candidate_list), candidate_list.shape, len(candidate_list), candidate_list)
                            else:
                                # print("candidate 2 before: ", candidate_list)
                                candidate_list = self.node_r_niegh_array[s_n][temp_r]
                                # print("candidate_list 2: ", type(candidate_list), candidate_list.shape, candidate_list)

                            # n_vector.extend(candidate_list[np.random.randint(0, len(candidate_list), r_sample)]) # 使用numpy会使得不同线程采样相同
                            if len(candidate_list) >= r_sample:
                                r_index = random.sample(range(0, len(candidate_list)), r_sample)
                            else:
                                n_repeat = r_sample // len(candidate_list) + 1
                                # print("range: ", list(range(0, len(candidate_list))))
                                n_list = list(range(0, len(candidate_list))) * n_repeat
                                # print("n_repeat: ", n_repeat)
                                # print("n_list: ", n_list)
                                r_index = random.sample(n_list, r_sample)
                            n_vector.extend(candidate_list[r_index])
                    # print("t output vector: ", n_vector)
                    # print("t output mask: ", n_mask)
                    t_vector.extend(n_vector)
                    t_mask.extend(n_mask)
                    t_r.extend(n_r)
                o_vector.append(np.asarray(t_vector, dtype=np.int64).reshape(input_size, -1))
                o_mask.append(np.asarray(t_mask, dtype=np.int32).reshape(input_size, -1))
                o_r.append(np.asarray(t_r, dtype=np.int8).reshape(input_size, -1))
            if len(o_vector[0].shape) == 1:
                o_vector = np.concatenate(o_vector)
                o_mask = np.concatenate(o_mask)
                o_r = np.concatenate(o_r)
            else:
                o_vector = np.concatenate(o_vector, axis=1)
                o_mask = np.concatenate(o_mask, axis=1)
                o_r = np.concatenate(o_r, axis=1)

            o_mask = o_mask.reshape((-1, self.max_neigh_r))
            o_r = o_r.reshape((-1, self.max_neigh_r))
            a, b = o_mask.shape
            c = self.sample_vector[-1]
            o_mask_dense = np.zeros((a, b, c), dtype=np.float16)
            for i in range(a):
                index = 0
                for j in range(b):
                    if o_mask[i, j]:
                        o_mask_dense[i, j, index:index + o_mask[i, j]] = 1.0 / o_mask[i, j]
                        index += o_mask[i, j]

            return o_vector, o_mask_dense.reshape(a, -1), o_r

        def _get_feed_dict(self, index):
            target_item = self.data['item_id'][index]
            neg_items = self.neg_items[index]
            item_ids = np.concatenate([[target_item], neg_items])
            feed_dict = {'item_id': item_ids}

            feed_dict['user_id'] = self.data['user_id'][index]

            # feed_dict['item_id'] = feed_dict['item_id'] - self.corpus.min_item
            feed_source_data = np.concatenate([[feed_dict['user_id']], feed_dict['item_id']])
            feed_dict['neigh'], feed_dict['mask'], feed_dict['r'] = self.get_neighbor(feed_source_data)
            # feed_dict['user_neigh'], feed_dict['user_mask'], feed_dict['user_r'] = self.get_neighbor(feed_dict['user_id'])
            # feed_dict['item_neigh'], feed_dict['item_mask'], feed_dict['item_r']  = self.get_neighbor(feed_dict['item_id'])
            return feed_dict


        def print_info(self, sp):
            indices = np.vstack((sp.row, sp.col)).astype(np.int64)
            data = sp.data.astype(np.double)
            size = sp.shape
            print("indices: {}".format(indices))
            print("data: {}".format(data))
            print("size: {}".format(size))


        def collate_batch(self, feed_dicts):
            feed_dict = dict()
            for tag in feed_dicts[0]:
                npy_t = np.array([t[tag] for t in feed_dicts])
                feed_dict[tag] = torch.from_numpy(npy_t)

            feed_dict['neg_item'] = len(feed_dicts[0]["item_id"])
            feed_dict['batch_size_rec'] = len(feed_dicts)
            feed_dict['phase'] = self.phase
            # print('batch_size: ', feed_dict['batch_size'])
            # print('phase: ', feed_dict['phase'])
            return feed_dict