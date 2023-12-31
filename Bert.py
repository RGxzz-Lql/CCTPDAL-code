import torch
import torch.nn as nn
#from pytorch_pretrained import BertModel, BertTokenizer
from pytorch_pretrained_bert import BertModel,BertTokenizer
from IPython import embed

class Config(object):
    """
    配置参数
    """
    def __init__(self, dataset):
        self.model_name = 'Bert'
        # 训练集
        self.train_path = dataset + '/data/train.txt'
        # 测试集
        self.test_path = dataset + '/data/test.txt'
        # 校验集
        self.dev_path = dataset + '/data/dev.txt'
        # dataset
        self.datasetpkl = dataset + '/data/datasetactive0829.pkl'
        # 类别
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        #模型训练结果
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        # 设备配置
        self.device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')

        # 若超过1000bacth效果还没有提升，提前结束训练
        self.require_improvement = 1000

        # 类别数
        self.num_classes = len(self.class_list)
        # epoch数
        self.num_epochs = 2
        # batch_size
        self.batch_size = 64
        # 每句话处理的长度(短填，长切）
        self.pad_size = 1
        # 学习率
        self.learning_rate = 1e-3
        # bert预训练模型位置
        self.bert_path = './bert_pretrain'
        # bert切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # bert隐层层个数
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        #print(config.num_classes)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # x [ids, seq_len, mask]
        context = x[0] #对应输入的句子 shape[32,128]
        mask = x[2] #对padding部分进行mask shape[32，128]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=True) #shape [32,768]
       #embed()
        out = self.fc(pooled) # shape [32,10]
        return out