# -*- coding: UTF-8 -*-

import os
import sys
import pickle
import logging
import argparse
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

from models import *
from helpers import *
from utils import utils
import pynvml

def parse_global_args(parser):
    parser.add_argument('--gpu', type=str, default='2',
                        help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--model_name', type=str, default='BPR',
                        help='Choose a model to run.')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--random_seed', type=int, default=2019,
                        help='Random seed of numpy and pytorch.')
    parser.add_argument('--load', type=int, default=0,
                        help='Whether load model and continue to train')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--regenerate', type=int, default=0,
                        help='Whether to regenerate intermediate files.')
    return parser


def main():
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
               'regenerate', 'sep', 'train', 'verbose']
    logging.info(utils.format_arg_str(args, exclude_lst=exclude))

    # Random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    # GPU
    logging.info("# args.gpu: {}".format(args.gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info("# cuda devices: {}".format(torch.cuda.device_count()))
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(args.gpu)) #默认用一张卡
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    logging.info("# cuda mem used: {}".format(meminfo.used))

    # Read data
    corpus_path = os.path.join(args.path, args.dataset, model_name.reader + '.pkl')
    if not args.regenerate and os.path.exists(corpus_path):
        logging.info('Load corpus from {}'.format(corpus_path))
        corpus = pickle.load(open(corpus_path, 'rb'))
        sep = corpus.sep
        prefix = corpus.prefix
        dataset = corpus.dataset
        history_max = corpus.history_max
        item_meta_df = corpus.data_df
        item_meta_df = corpus.item_meta_df
    else:
        corpus = reader_name(args)
        logging.info('Save corpus to {}'.format(corpus_path))
        pickle.dump(corpus, open(corpus_path, 'wb'))
        sep = corpus.sep
        prefix = corpus.prefix
        dataset = corpus.dataset
        history_max = corpus.history_max
        item_meta_df = corpus.data_df
        item_meta_df = corpus.item_meta_df
    # Define model
    model = model_name(args, corpus)
    logging.info(model)

    # model = model.double()

    model.apply(model.init_weights)
    model.actions_before_train()
    if torch.cuda.device_count() > 0:
        model = model.cuda()

    # Run model
    data_dict = dict()
    for phase in ['dev', 'test']:
        data_dict[phase] = model_name.Dataset(model, corpus, phase)
    print("Model_name: ", args.model_name)
    if args.model_name.find("KG") >= 0 or args.model_name == "LegalGNN":
        data_dict["train"] = {}
        data_dict["train"]["rec"] = model_name.Dataset(model, corpus, "train")
        data_dict["train"]["kg"] = model_name.Dataset_kg(model, corpus, "train")
    else:
        data_dict["train"] = model_name.Dataset(model, corpus, "train")

    runner = runner_name(args)

    for phase in ['dev', 'test']:
        print("loading data")
        start_t = time.time()
        data_list = []
        dl = DataLoader(data_dict[phase], batch_size=runner.eval_batch_size, shuffle=False, num_workers=runner.num_workers,
                        collate_fn=lambda x: x, pin_memory=runner.pin_memory)
        for batch in tqdm(dl, leave=False, desc='Pre data', ncols=100, mininterval=1):
            data_list.extend(batch)
        data_dic = {}
        for i, d in enumerate(data_list):
            data_dic[i] = d
        data_dict[phase].buffer_dict = data_dic
        print("load done, spend time: ", time.time() - start_t)
        print("data_len: ", len(data_dic))

    logging.info('Test Before Training: ' + runner.print_res(model, data_dict['test']))
    if args.load > 0:
        model.load_model()
    if args.train > 0:
        runner.train(model, data_dict)
    logging.info(os.linesep + 'Test After Training: ' + runner.print_res(model, data_dict['test']))
    logging.info(os.linesep + 'Valid After Training: ' + runner.print_res(model, data_dict['dev']))

    model.actions_after_train()
    logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='BPR', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    model_name = eval('{0}.{0}'.format(init_args.model_name))

    print(model_name.reader)
    print(model_name.reader)
    print(model_name.runner)
    reader_name = eval('{0}.{0}'.format(model_name.reader))
    runner_name = eval('{0}.{0}'.format(model_name.runner))
    print(type(reader_name))


    # Args
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = reader_name.parse_data_args(parser)
    parser = runner_name.parse_runner_args(parser)
    parser = model_name.parse_model_args(parser)
    args, extras = parser.parse_known_args()

    # Logging configuration
    log_args = [init_args.model_name, args.dataset, str(args.random_seed)]
    for arg in ['lr', 'l2'] + model_name.extra_log_args:
        log_args.append(arg + '=' + str(eval('args.' + arg)))
    log_file_name = '__'.join(log_args).replace(' ', '__')
    if args.log_file == '':
        args.log_file = '../log/{}/{}.txt'.format(init_args.model_name, log_file_name)
    if args.model_path == '':
        args.model_path = '../model/{}/{}.pt'.format(init_args.model_name, log_file_name)
    print(args.log_file)
    utils.check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(init_args)

    main()
