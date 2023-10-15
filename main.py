import time
import torch
import numpy as np
from importlib import import_module
import argparse
import utils
import train




parser = argparse.ArgumentParser(description='Bruce-Bert-Text-Classsification')
parser.add_argument('--model', type=str, default='Bert', help = 'choose a model Bert, BertCNN, BertRNN, BertDPCNN,BertRCNN.')
args = parser.parse_args()



if __name__ == '__main__':
    #dataset = 'data3600'  # 数据集地址
    #datasets = ['data200','data900','data1800','data3600','data7200','data10800','data14400',]
    #datasets = [ 'data3600', 'data7200', 'data10800', 'data14400', ]
    datasets = ['data18000']
    # 数据集地址
    for dataset in  datasets:
        print(args.model)
        print(dataset)
        print(args.model)
        model_name = args.model
        model = import_module('model.' + model_name)
        #print(model)
        config = model.Config(dataset)
        #print(config)
        model = model.Model(config).to(config.device)
        # model.load_state_dict(torch.load(config.save_path))
        # model.eval()
        ##

        start_time = time.time()
        print('加载数据集')
        train_data, dev_data, test_data = utils.bulid_dataset(config)
        # print(test_data)


        ##

        # print(test_data)
        train_iter = utils.bulid_iterator(train_data, config)
        dev_iter = utils.bulid_iterator(dev_data, config)
        test_iter = utils.bulid_iterator(test_data, config, )

        # model = config.Model(config).to(config.device)
        train.train(config, model, train_iter, dev_iter, test_iter, args.model, dataset)