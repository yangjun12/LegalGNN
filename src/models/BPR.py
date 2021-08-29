# -*- coding: UTF-8 -*-

import torch

from models.BaseModel import BaseModel
import numpy as np


class BPR(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        return BaseModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.user_num = corpus.n_users
        super().__init__(args, corpus)

    def _define_params(self):
        self.u_embeddings = torch.nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = torch.nn.Embedding(self.item_num, self.emb_size)
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)

    # @torchsnooper.snoop()
    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]

        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)
        u_bias = self.user_bias(u_ids)
        i_bias = self.item_bias(i_ids).squeeze(-1)


        prediction = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(dim=-1)  # [batch_size, -1]
        prediction = prediction + u_bias + i_bias
        return prediction.view(feed_dict['batch_size'], -1)

    class Dataset(BaseModel.Dataset):
        def _get_feed_dict(self, index):
            target_item = self.data['item_id'][index]
            neg_items = self.neg_items[index]
            item_ids = np.concatenate([[target_item], neg_items])
            user_ids = self.data['user_id'][index]
            # made item: origin range[min_item, max_item] => range[0, n_items]
            item_ids = item_ids - self.corpus.min_item

            feed_dict = {'item_id': item_ids, 'user_id': user_ids}
            return feed_dict
