# -*- coding: UTF-8 -*-

import os
import time
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.sparse as sp


class LegalGNNReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='../data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        parser.add_argument('--history_max', type=int, default=20,
                            help='Maximum length of history.')
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self.history_max = args.history_max

        t0 = time.time()
        self._load_data()
        self._append_info()
        logging.info('Done! [{:<.2f} s]'.format(time.time() - t0) + os.linesep)

    def _load_data(self):
        logging.info('Loading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df, self.item_meta_df = dict(), pd.DataFrame()
        self._load_preprocessed_df()

        logging.info('Counting dataset statistics...')
        # self.all_df = pd.concat([df[['user_id', 'item_id', 'time']] for df in self.data_df.values()])

        self.n_clicks = len(self.data_df["train"])
        # self.min_time, self.max_time = self.all_df['time'].min(), self.all_df['time'].max()
        self.n_relations = self.relation_df['relation'].max() + 1
        self.n_triplets = len(self.relation_df)

        self._load_sparse_feature()
        self._load_dense_feature()

        logging.info('"# users": {}, "# items": {}, "# clicks": {}'.format(self.n_users, self.n_items, self.n_clicks))
        logging.info('"# relations": {}, "# triplets": {}'.format(self.n_relations, self.n_triplets))

    def _append_info(self):
        """
        Add history info to data_df: item_his, time_his, his_length
        ! Need data_df to be sorted by time in ascending order
        :return:
        """

        user_clicked_list = defaultdict(list)
        for uid, iid in zip(self.data_df["train"]["user_id"].tolist(), self.data_df["train"]["item_id"].tolist()):
            user_clicked_list[uid].append(iid)
        self.user_clicked_set = {}
        for uid in user_clicked_list:
            self.user_clicked_set[uid] = set(user_clicked_list[uid])

    def _load_sparse_feature(self):
        all_item_rows = []
        all_item_cols = []

        item_rows = []
        item_cols = []

        self.neigh_r = defaultdict(dict)
        self.neigh = defaultdict(list)
        print("min_item {} {}".format(type(self.min_item), self.min_item))
        print("max_item {} {}".format(type(self.max_item), self.max_item))
        for head, relation, tail in zip(self.relation_df['head'].tolist(), self.relation_df['relation'].tolist(),
                                        self.relation_df['tail'].tolist()):
            all_item_rows.append(head)
            all_item_cols.append(tail)
            self.neigh[head].append(tail)
            self.neigh[tail].append(head)
            if relation not in self.neigh_r[head]:
                self.neigh_r[head][relation] = [tail]
            else:
                self.neigh_r[head][relation].append(tail)
            r_relation =relation + self.n_relations
            if r_relation not in self.neigh_r[tail]:
                self.neigh_r[tail][r_relation] = [head]
            else:
                self.neigh_r[tail][r_relation].append(head)
            if head < self.max_item and head > self.min_item:
                item_rows.append(head)
                item_cols.append(tail)
        print("item_cols.length: {}".format(len(item_cols)))
        all_item = list(range(self.min_item, self.max_item))
        item_rows.extend(all_item)
        item_cols.extend(all_item)
        cat_values = [1.] * len(item_rows)
        self.item_sparse_feature = sp.coo_matrix((cat_values, (item_rows, item_cols)), shape=(self.n_entity, self.n_entity)).tocsr()

        all_entity = list(range(0, self.n_entity))
        all_item_cols.extend(all_entity)
        all_item_rows.extend(all_entity)
        all_cat_values = [1.] * len(all_item_cols)
        self.all_sparse_feature = sp.coo_matrix((all_cat_values, (all_item_rows, all_item_cols)), shape=(self.n_entity, self.n_entity)).tocsr()

        for e in self.neigh:
            self.neigh[e] = list(set(self.neigh[e]))
        for e in self.neigh_r:
            for r in self.neigh_r[e]:
                self.neigh_r[e][r] = list(set(self.neigh_r[e][r]))

        user_rows = []
        user_cols = []
        all_user = list(range(0, self.min_item))
        user_rows.extend(all_user)
        user_cols.extend(all_user)
        user_values = [1.] * len(all_user)
        self.user_sparse_feature = sp.coo_matrix((user_values, (user_rows, user_cols)), shape=(self.n_entity, self.n_entity)).tocsr()
    def _load_dense_feature(self):

        # fake feature
        self.item_dense_feature = np.random.rand(self.n_items, 64)

    def _load_preprocessed_df(self):
        # item_meta_path = os.path.join(self.prefix, self.dataset, 'item_meta.csv')
        # if os.path.exists(item_meta_path):
        #     self.item_meta_df = pd.read_csv(item_meta_path, sep=self.sep)
        # self.data_df['train'] = pd.read_csv(os.path.join(self.prefix, self.dataset, 'train.csv'), sep=self.sep)
        # self.data_df['dev'] = pd.read_csv(os.path.join(self.prefix, self.dataset, 'dev.csv'), sep=self.sep)
        # self.data_df['test'] = pd.read_csv(os.path.join(self.prefix, self.dataset, 'test.csv'), sep=self.sep)
        node_type_dic = {}
        with open(os.path.join(self.prefix, self.dataset, 'node_type_map.txt')) as fin:
            data = fin.readlines()
            for l in data:
                line = l.rstrip("\n").split(" ")
                assert(len(line) == 2)
                assert(len(line) == 2)
                node_type_dic[line[0]] = int(line[1])
                logging.info("{} : {}".format(line[0], line[1]))

        entity_df = pd.read_csv(os.path.join(self.prefix, self.dataset, "entity_list.txt"), header=None, sep="\t")
        entity_df.columns = ["origin_id", "type", "entity_id"]

        user_df = entity_df[entity_df["type"] == node_type_dic["user"]]
        item_df = entity_df[entity_df["type"] == node_type_dic["item"]]
        query_df = entity_df[entity_df["type"] == node_type_dic["query"]]
        min_query, max_query = query_df["entity_id"].min(), query_df["entity_id"].max() + 1
        self.min_item = item_df["entity_id"].min(0)
        self.max_item = item_df["entity_id"].max(0) + 1 # min_item <= item_id < max_item
        self.n_entity = entity_df["entity_id"].max(0) + 1
        self.n_users = self.min_item
        self.n_items = self.max_item - self.min_item

        self.node_type_dic = node_type_dic # {"user": 0}
        self.node_type_list = entity_df["type"].tolist() # [0,...0,1,...,1,....]


        logging.info("n_entity: {} n_users: {} n_items: {}".format(self.n_entity, self.n_users, self.n_items))
        logging.info("user range: [{}, {}], shape: {}".format(self.min_item, self.max_item, user_df.shape))
        logging.info("item range: [{}, {}], shape: {}".format(item_df["entity_id"].min(), item_df["entity_id"].max() + 1, item_df.shape))
        logging.info("query range: [{}, {}], shape: {}".format(query_df["entity_id"].min(), query_df["entity_id"].max() + 1, item_df.shape))
        self.relation_df = pd.read_csv(os.path.join(self.prefix, self.dataset, 'kg_final.txt'), header=None, sep=" ")
        self.relation_df.columns = ["head", "relation", "tail"]
        print(self.relation_df.describe())
        print(self.relation_df.shape)
        print(pd.__version__)
        logging.info("relation orgin: {}".format(len(self.relation_df["head"])))

        logging.info("shape of relation_df: {}".format(self.relation_df.shape))

        self.triplet_set = set()

        for head, relation, tail in zip(self.relation_df['head'].tolist(), self.relation_df['relation'].tolist(),
                                        self.relation_df['tail'].tolist()):
            self.triplet_set.add((head, relation, tail))

        def read_file(file_name):
            data_list = []
            with open(file_name) as fin:
                data = fin.readlines()
                for l in data:
                    line = l.rstrip("\n").split("\t")
                    data_list.append(line)
            return data_list

        data_train = read_file(os.path.join(self.prefix, self.dataset, 'train_h.txt'))
        train_user, train_item, history_items, history_actions = [], [], [], []
        for a, b, c, d in data_train:
            train_user.append(int(a))
            train_item.append(int(b))
            history_items.append([int(t) for t in c.split(" ") if t])
            history_actions.append([int(t) for t in d.split(" ") if t])
            print(a, b, "#", [int(t) for t in c.split(" ") if t])
        self.data_df["train"] = pd.DataFrame()
        self.data_df["train"]["user_id"] = train_user
        self.data_df["train"]["item_id"] = train_item
        self.data_df["train"]["hist_id"] = history_items
        self.data_df["train"]["hist_actions"] = history_items

        def get_test_data(file_name):
            data_test = read_file(file_name)
            test_user, test_item, test_neg_item, history_items, history_actions = [], [], [], [], []
            for a, b, c, d, e in data_test:
                test_user.append(int(a))
                test_item.append(int(b))
                test_neg_item.append([int(t) for t in c.split(" ")[1:]])
                history_items.append([int(t) for t in d.split(" ") if t])
                history_actions.append([int(t) for t in e.split(" ") if t])
            test_df = pd.DataFrame()
            test_df["user_id"] = test_user
            test_df["item_id"] = test_item
            test_df["neg_items"] = test_neg_item
            test_df["hist_id"] = history_items
            test_df["hist_actions"] = history_actions
            return test_df

        self.data_df["test"] = get_test_data(os.path.join(self.prefix, self.dataset, 'test.txt'))
        self.data_df["dev"] = get_test_data(os.path.join(self.prefix, self.dataset, 'valid.txt'))
