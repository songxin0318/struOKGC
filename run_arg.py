import json
import os
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
import sys
from os.path import join, abspath, dirname
from eval_utils import *
from data_utils.dataset import *
from data_utils.utils import *
from prompt_tuning.modeling import PTuneForLAMA

'''
模型的主体部分
'''
#support models列表  给出了一些用到的PLM的模型名称
SUPPORT_MODELS = ['bert-base-cased', 'bert-large-cased', 'bert-base-uncased','sapbert']

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 :
        torch.cuda.manual_seed_all(args.seed)

#参数的设置
def construct_generation_args():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    # parser.add_argument("--model_name", type=str, default='sapbert', choices=SUPPORT_MODELS)
    parser.add_argument("--model_name", type=str, default='bert-base-cased', choices=SUPPORT_MODELS)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')   #伪token

    parser.add_argument("--t5_shard", type=int, default=0)
    parser.add_argument("--template", type=str, default="(1,1,1,1,1,1)")
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--valid_step", type=int, default=10000)
    parser.add_argument("--recall_k", type=int, default=30)
    parser.add_argument("--pos_K", type=int, default=30)
    parser.add_argument("--neg_K", type=int, default=30)
    parser.add_argument("--random_neg_ratio", type=float, default=0.5)
    parser.add_argument("--keg_neg", type=str, default='all', choices=['all', 'tail'])

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lm_lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=234, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # lama configuration
    parser.add_argument("--use_original_template", action='store_true')
    parser.add_argument("--use_lm_finetune", action='store_true')

    parser.add_argument("--link_prediction", action='store_true',default=True)
    parser.add_argument("--output_cla_results", action='store_true',default=True)
    parser.add_argument("--add_definition", action='store_true')
    parser.add_argument("--test_open", action='store_true')

    # directories  路经参数
    # parser.add_argument("--data_dir", type=str, default='./dataset/UMLS-PubMed')
    parser.add_argument("--data_dir", type=str, default='./dataset/wiki27K')
    parser.add_argument("--out_dir", type=str, default='./checkpoint/wiki27K')
    parser.add_argument("--load_dir", type=str, default='')


    args = parser.parse_args()

    # post-parsing args
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template

    assert type(args.template) is tuple

    set_seed(args)
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device='cuda:0'   # self.device = 'cuda:0'
        # load tokenizer
        tokenizer_src = self.args.model_name

        if self.args.model_name == 'kepler':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        # elif self.args.model_name == 'luke':
        #     self.tokenizer = LukeTokenizer.from_pretrained('studio-ousia/luke-base')
        elif self.args.model_name == 'sapbert':
            self.tokenizer = AutoTokenizer.from_pretrained(
                'SapBERT-from-PubMedBERT-fulltext')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            print('run_arg.py tokenizer load  over!!')
            #wiki27k数据集,用了roberta-large数据集  定义了一个PLM的分词器.

        os.makedirs(self.get_save_path(), exist_ok=True)   #生成输出模型的路径
        # get_dataloader数据加载
        # 传入tokenizer分词器,作为参数
        # util包
        self.train_loader, self.dev_loader, self.test_loader, self.ch_test_loader, \
                            self.oh_test_loader, self.o_test_loader, self.link_loader_head, \
                            self.link_loader_tail, relation_num, self.link_dataset_head, self.link_dataset_tail, self.config_sx, tokenizer_str \
                            = get_dataloader(args, self.tokenizer)
        #运行完成,会输出"文件加载完毕"
        print('sx -数据加载结束...')
        #定义prompt 微调模型
        self.model = PTuneForLAMA(args, self.device, self.args.template, self.tokenizer, relation_num,self.config_sx)   # modeling 包
        if self.args.load_dir != '':
            self.load(self.args.load_dir)

    def get_task_name(self):
        str_template = [str(x) for x in self.args.template]
        str_template = '.'.join(str_template)
        names = [self.args.model_name,
                 "template_{}".format(str_template),
                 "seed_{}".format(self.args.seed)]
        return "_".join(names)

    def get_save_path(self):
        return join(self.args.out_dir, self.args.model_name, 'search', self.get_task_name())

    def get_checkpoint(self, epoch_idx, dev_f1, test_f1):
        ckpt_name = "epoch_{}_dev_{}_test_{}.ckpt".format(epoch_idx, round(dev_f1 * 100, 4),
                                                          round(test_f1 * 100, 4))
        return {'model_state_dict': self.model.state_dict(),
                'ckpt_name': ckpt_name,
                'dev_f1': dev_f1,
                'test_f1': test_f1}

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))
        print("# Checkpoint {} saved.".format(ckpt_name))

    def load(self, load_path):
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def train(self):
        best_dev, early_stop, has_adjusted = 0, 0, True
        best_ckpt = None
        params = [{'params': self.model.prompt_encoder.parameters()}]
        if self.args.use_lm_finetune:
            params.append({'params': self.model.model.parameters(), 'lr': self.args.lm_lr})
        optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        print("start epoch ---------")
        for epoch_idx in range(self.args.max_epoch):
            print("-----------epoch_idx: ",epoch_idx)  #进入每一次的epoch
            # run training
            pbar = tqdm(self.train_loader)    #加载self.train_loader
            # for batch_idx, batch in enumerate(self.train_loader):

            for batch_idx, batch in enumerate(pbar):  #遍历train_loader中每个batch的数据
                # print('*******')
                self.model.train()
                # bath[0] batch[1] batch[2] 分别指的是什么??  batch_size 提前设定为16设定
                # print(batch[0])  # texts
                # print(batch[1])  # rs
                # print(batch[2])  #labels
                # print(len(batch)) #5  texts, rs, labels,tripls(暂时用这个), global_is_sx(暂时不用)
                # print(batch[3])  #6三元组list  ['h\r\t','',...],根据这个batch 获得邻居
                # loss, acc, _ = self.model.forward_classification(batch[0], batch[1], batch[2])  #
                #sx修改之后,每个batch是一个字典
                texts=batch['triple_text']
                rs=batch['rs']
                labels=batch['label']

                #struc_embedding encoder部分的输入
                batch_neighbors={}
                batch_neighbors['struc_data']=batch['struc_data']
                batch_neighbors['struc_neighbor']=batch['struc_neighbor']
                batch_neighbors['neighbors_label']=batch['neighbors_label']

                loss, acc, _ = self.model.forward_classification(texts, rs, labels,batch_neighbors)  #
                # print('self.model.forward_classification over over over')
                # loss, acc, _ = self.model.forward_classification(batch[0], batch[1], batch[2],batch[3])  #
                pbar.set_description(f"Loss {float(loss.mean()):.6g}, acc {acc:.4g}")

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # check early stopping
                if batch_idx % self.args.valid_step == 0:  #初始化,self.args.valid_step=10000
                    # Triple Classification #三元组分类 sx 需要和train 对应起来
                    #返回验证集和测试集的F1值
                    dev_results, test_results = evaluate_classification_using_classification(self, epoch_idx)
                    print('evaluate_classification_using_classification  over')


                    # Link Prediction   #sx 三元组链接预测   link_prediction bool 参数控制是否进行链接预测任务
                    #条件成立  self.args.link_prediction为true,  且  batch_idx不是0,epoch_idx不为0
                    if self.args.link_prediction and not (batch_idx == 0 and epoch_idx == 0):  #and 前后均为true
                        print('sx-link-prediction',self.args.link_prediction,batch_idx,epoch_idx)
                        evaluate_link_prediction_using_classification(self, epoch_idx, batch_idx, output_scores=True)

                    # Early stop and save
                    if dev_results >= best_dev:
                        best_ckpt = self.get_checkpoint(epoch_idx, dev_results, test_results)
                        early_stop = 0
                        best_dev = dev_results
                    if False:
                        print('False')

                    else:
                        early_stop += 1
                        if early_stop >= self.args.early_stop:
                            self.save(best_ckpt)
                            print("Early stopping at epoch {}.".format(epoch_idx))
                            print("FINISH_TRAIN...")
                            return best_ckpt

                    sys.stdout.flush()
            my_lr_scheduler.step()
        self.save(best_ckpt)

        return best_ckpt


def main():

    args = construct_generation_args()
    print("cuda", torch.cuda.is_available())

    if type(args.template) is not tuple:
        args.template = eval(args.template)
    assert type(args.template) is tuple
    print('sx00 args.model_name: ',args.model_name)

    trainer = Trainer(args)  #  初始换构建trainer
    print('sx01', trainer)
    print('sx02 trainer.model.template:',trainer.model.template)

    print("sx11  start training-------")
    trainer.train()


if __name__ == '__main__':
    main()
