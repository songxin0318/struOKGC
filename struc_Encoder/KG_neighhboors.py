import os
import copy
import json
import shutil
from time import strftime, localtime
import numpy as np
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import argparse

# nothing special in this class
from transformers import BertTokenizer


class KGCDataset(Dataset):
    def __init__(self, data: list):
        super(KGCDataset, self).__init__()

        self.data = data
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]


# prepare the dataset
class KGCDataModule:
    def __init__(self, args: dict, tokenizer, encode_text=False, encode_struc=False):
        # 0. some variables used in this class
        self.task = args['task']
        self.data_path = args['data_path']
        self.batch_size = args['batch_size']
        self.num_workers = args['num_workers']
        self.pin_memory = args['pin_memory']
        self.max_seq_length = args['max_seq_length'] if encode_text else -1

        self.add_neighbors = args['add_neighbors']
        self.neighbor_num = args['neighbor_num']
        self.neighbor_token = args['neighbor_token']
        self.no_relation_token = args['no_relation_token']

        self.encode_text = encode_text #  文本编码器
        self.encode_struc = encode_struc# 结构编码器

        # 1. read entities and relations from files
        # self.entities, self.relations = self.read_support()  # 获取 实体字典 和  关系字典
        self.entities, self.relations = self.read_support_wiki27k_sx()   #获取 实体字典 和  关系字典
        print(f'Number of entities: {len(self.entities)}; Number of relations: {len(self.relations)}') #输出 实体和关系数量

        # 2.1 expand the tokenizer for BERT
        self.tokenizer = tokenizer

        text_offset = self.resize_tokenizer()   #这个函数是基于self.entities 和 self.relations进行操作的,
        # 返回是4项,实体开始 结束token_idx  以及 关系的开始 结束token_idx  这个时候self tokenizen变了，增加了词汇表。
        # self.tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
        # self.pseudo_token_id = self.tokenizer.get_vocab()['[PROMPT]']

        # 2.2 construct the vocab for our KGE module
        self.vocab, struc_offset = self.get_vocab()

        # 2.3 offsets indicate the positions of entities in the vocab, we store them in args and pass to other classes
        args.update(text_offset)
        args.update(struc_offset)
        # 2.4 the following two variables will be used to construct the KGE module
        args['vocab_size'] = len(self.vocab)
        args['num_relations'] = struc_offset['struc_relation_end_idx'] - struc_offset['struc_relation_begin_idx']

        # 3.1 read the dataset
        self.lines = self.read_lines()  # {'train': [(h,r,t),...], 'dev': [], 'test': []}
        # 3.2 use the training set to get neighbors
        self.neighbors = self.get_neighbors()  # {ent: {text_prompt: [], struc_prompt: []}, ...}

        # 3.3 entities to be filtered when predict some triplet
        self.entity_filter = self.get_entity_filter()

        # 5. use the triplets in the dataset to construct the inputs for our BERT and KGE module
        if self.task == 'pretrain':
            # utilize entities to get dataset when task is pretrain
            examples = self.create_pretrain_examples()
        else:
            examples = self.create_examples()


        # 6. the above inputs are used to construct pytorch Dataset objects
        self.train_stru=examples['train']
        self.dev_stru = examples['dev']
        self.test_stru = examples['test']
        # self.train_ds = KGCDataset(examples['train'])     #
        # self.dev_ds = KGCDataset(examples['dev'])
        # self.test_ds = KGCDataset(examples['test'])

    # the following six functions are called in the __init__ function
    #获取 实体词典 和  关系词典
    #根据 PKGC的数据格式,修改完毕
    #接收的参数是,self.data_path  数据集所在路径

    def read_support(self):
        """
        read entities and relations from files
        :return: two Python Dict objects
        """
        entity_path = os.path.join(self.data_path, 'support', 'entity.json')  #存储实体信息的json文件路径
        entities = json.load(open(entity_path, 'r', encoding='utf-8'))
        for idx, e in enumerate(entities):  # 14541
            new_name = f'[E_{idx}]'
            raw_name = entities[e]['name']
            desc = entities[e]['desc']
            entities[e] = {
                'token_id': idx,  # used for filtering
                'name': new_name,  # new token to be added in tokenizer because raw name may consist many tokens
                'desc': desc,  # entity description, which improve the performance significantly
                'raw_name': raw_name,  # meaningless for the model, but can be used to print texts for debugging
            }

        relation_path = os.path.join(self.data_path, 'support', 'relation.json')
        relations = json.load(open(relation_path, 'r', encoding='utf-8'))
        for idx, r in enumerate(relations):  # 237
            sep1, sep2, sep3, sep4 = f'[R_{idx}_SEP1]', f'[R_{idx}_SEP2]', f'[R_{idx}_SEP3]', f'[R_{idx}_SEP4]'
            name = relations[r]['name']
            relations[r] = {
                'sep1': sep1,  # sep1 to sep4 are used as soft prompts
                'sep2': sep2,
                'sep3': sep3,
                'sep4': sep4,
                'name': name,  # raw name of relations, we do not need new tokens to replace raw names
            }

        return entities, relations

    #适配PKGC模型的数据读取函数
    def read_support_wiki27k_sx(self):
        """
        read entities and relations from files
        :return: two Python Dict objects
        """
        entity_label_path = os.path.join(self.data_path, 'support', 'entity2label.txt')  # 存储实体信息的json文件路径
        # entities_label = json.load(open(entity_label_path, 'r', encoding='utf-8'))
        entities_label_lines = open(entity_label_path).readlines()
        entity_defi_path = os.path.join(self.data_path, 'support', 'entity2definition.txt')
        entities_defi_lines = open(entity_defi_path).readlines()

        print('---------------entities-------------')
        # print(type(entities))
        # print(entities)
        entities = {}
        i = 0
        for entity_line in entities_label_lines:
            entity, label = entity_line.strip().split('\t')
            new_name = f'[E_{i}]'  # 实体e的序号,就是按序号
            entities[entity] = {
                'token_id': i,  # used for filtering
                'name': new_name,  # new token to be added in tokenizer because raw name may consist many tokens
                'raw_name': label,  # meaningless for the model, but can be used to print texts for debugging
                            }
            i = i + 1
        # 获取实体的desc(从definition文件获取)
        for entity_line in entities_defi_lines:
            entity_defi, desc = entity_line.strip().split('\t')
            entities[entity_defi]['desc'] = desc

        # for idx, e in enumerate(entities):  # 14541  #idx是枚举的序号,0,1,2,...entity_num;  e 是实体的代号
        #     # print('entities-idx:',idx)
        #     # print('entities-e:', e)
        #     new_name = f'[E_{idx}]'   #实体e的序号,就是按序号
        #     raw_name = entities[e]['name']  #获取实体e的真正名称
        #     desc = entities[e]['desc'] #获取实体e的描述信息
        #     entities[e] = {
        #         'token_id': idx,  # used for filtering
        #         'name': new_name,  # new token to be added in tokenizer because raw name may consist many tokens
        #         'desc': desc,  # entity description, which improve the performance significantly
        #         'raw_name': raw_name,  # meaningless for the model, but can be used to print texts for debugging
        #     }

        # relation_path = os.path.join(data_path, 'support', 'relation.json')
        # relations = json.load(open(relation_path, 'r', encoding='utf-8'))
        relation_path_label = os.path.join(self.data_path, 'support', 'relation2label.json')
        # relations=open(relation_path_label).readlines()
        relations = json.load(open(relation_path_label, 'r', encoding='utf-8'))
        print('---------------relations-------------')
        # print(type(relations))
        # print(relations)
        for idx, r in enumerate(relations):  # 237
            sep1, sep2, sep3, sep4 = f'[R_{idx}_SEP1]', f'[R_{idx}_SEP2]', f'[R_{idx}_SEP3]', f'[R_{idx}_SEP4]'
            name = relations[r]
            relations[r] = {
                'sep1': sep1,  # sep1 to sep4 are used as soft prompts
                'sep2': sep2,
                'sep3': sep3,
                'sep4': sep4,
                'name': name,  # raw name of relations, we do not need new tokens to replace raw names
            }
        return entities, relations

    def resize_tokenizer(self):
        """
        add the new tokens in self.entities and self.relations into the tokenizer of BERT
        :return: a Python Dict, indicating the positions of entities in logtis
        """
        entity_begin_idx = len(self.tokenizer)
        entity_names = [self.entities[e]['name'] for e in self.entities]
        # num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': entity_names})
        entity_end_idx = len(self.tokenizer)

        relation_begin_idx = len(self.tokenizer)
        relation_names = [self.relations[r]['sep1'] for r in self.relations]
        relation_names += [self.relations[r]['sep2'] for r in self.relations]
        relation_names += [self.relations[r]['sep3'] for r in self.relations]
        relation_names += [self.relations[r]['sep4'] for r in self.relations]
        relation_names += [self.neighbor_token, self.no_relation_token]
        # num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': relation_names})
        relation_end_idx = relation_begin_idx + 4 * len(self.relations) + 2

        return {
            'text_entity_begin_idx': entity_begin_idx,
            'text_entity_end_idx': entity_end_idx,
            'text_relation_begin_idx': relation_begin_idx,
            'text_relation_end_idx': relation_end_idx,
        }


    def get_vocab(self):
        """
        construct the vocab for our KGE module
        :return: two Python Dict
        """
        tokens = ['[PAD]', '[MASK]', '[SEP]', self.no_relation_token]
        entity_names = [e for e in self.entities]
        relation_names = []
        for r in self.relations:
            relation_names += [r, f'{r}_reverse']

        entity_begin_idx = len(tokens)
        entity_end_idx = len(tokens) + len(entity_names)
        relation_begin_idx = len(tokens) + len(entity_names)
        relation_end_idx = len(tokens) + len(entity_names) + len(relation_names)

        tokens = tokens + entity_names + relation_names
        vocab = dict()
        for idx, token in enumerate(tokens):
            vocab[token] = idx

        return vocab, {
            'struc_entity_begin_idx': entity_begin_idx,
            'struc_entity_end_idx': entity_end_idx,
            'struc_relation_begin_idx': relation_begin_idx,
            'struc_relation_end_idx': relation_end_idx,
        }

    #读取三元组是会过滤掉没有文本信息的三元组
    def read_lines(self):
        """
        read triplets from  files
        :return: a Python Dict, {train: [], dev: [], test: []}
        """
        data_paths = {
            'train': os.path.join(self.data_path, 'train.txt'),
            'dev': os.path.join(self.data_path, 'dev.txt'),
            'test': os.path.join(self.data_path, 'test.txt')
        }

        lines = dict()
        for mode in data_paths:
            data_path = data_paths[mode]
            raw_data = list()

            # 1. read triplets from files
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    h, r, t = str(line).strip().split('\t')
                    raw_data.append((h, r, t))

            # 2. filter triplets which have no textual information
            data = list()
            for h, r, t in raw_data:
                if (h in self.entities) and (t in self.entities) and (r in self.relations):
                    data.append((h, r, t))
            if len(raw_data) > len(data):
                raise ValueError('There are some triplets missing textual information')
            lines[mode] = data

        return lines

    #sx get邻居
    def get_neighbors(self):
        """
        construct neighbor prompts from training dataset 为训练数据集构建
        :return: {entity_id: {text_prompt: [], struc_prompt: []}, ...}
        """
        sep_token = self.tokenizer.sep_token
        mask_token = self.tokenizer.mask_token

        lines = self.lines['train']
        data = {e: {'text_prompt': [], 'struc_prompt': []} for e in self.entities}
        for h, r, t in lines:
            head, rel, tail = self.entities[h], self.relations[r], self.entities[t]
            h_name, r_name, t_name = head['name'], rel['name'], tail['name']
            sep1, sep2, sep3, sep4 = rel['sep1'], rel['sep2'], rel['sep3'], rel['sep4']

            # 1. neighbor prompt for predicting head entity 考虑的是关系的逆关系
            head_text_prompt = f'{sep1} {mask_token} {sep2} {r_name} {sep3} {t_name} {sep4}'
            head_struc_prompt = [self.vocab[t], self.vocab[f'{r}_reverse'], self.vocab[mask_token]]
            data[h]['text_prompt'].append(head_text_prompt)
            data[h]['struc_prompt'].append(head_struc_prompt)
            # 2. neighbor prompt for predicting tail entity  考虑的是正常的三元组
            tail_text_prompt = f'{sep1} {h_name} {sep2} {r_name} {sep3} {mask_token} {sep4}'
            tail_struc_prompt = [self.vocab[h], self.vocab[r], self.vocab[mask_token]]
            data[t]['text_prompt'].append(tail_text_prompt)
            data[t]['struc_prompt'].append(tail_struc_prompt)

        # add a fake neighbor if there is no neighbor for the entity
        for ent in data:
            if len(data[ent]['text_prompt']) == 0:
                h_name = self.entities[ent]['name']
                text_prompt = ' '.join([h_name, sep_token, self.no_relation_token, sep_token, mask_token])
                struc_prompt = [self.vocab[ent], self.vocab[self.no_relation_token], self.vocab[mask_token]]
                data[ent]['text_prompt'].append(text_prompt)
                data[ent]['struc_prompt'].append(struc_prompt)

        return data

    def get_entity_filter(self):
        """
        for given h, r, collect all t
        :return: a Python Dict, {(h, r): [t1, t2, ...]}
        """
        train_lines = self.lines['train']
        dev_lines = self.lines['dev']
        test_lines = self.lines['test']
        lines = train_lines + dev_lines + test_lines

        entity_filter = defaultdict(set)
        for h, r, t in lines:
            entity_filter[h, r].add(self.entities[t]['token_id'])
            entity_filter[t, r].add(self.entities[h]['token_id'])
        return entity_filter

    def create_examples(self):
        """
        :return: {train: [], dev: [], test: []}
        """
        examples = dict()
        for mode in self.lines:
            # print('mode',mode)  #train/dev/test
            data = list()
            lines = self.lines[mode]
            for h, r, t in tqdm(lines, desc=f'[{mode}]构建examples'):
                # print(h, r, t)
                head_example, tail_example = self.create_one_example(h, r, t)
                data.append(head_example)
                data.append(tail_example)
            examples[mode] = data
        print('create_examples')
        return examples

    def create_one_example(self, h, r, t):
        mask_token = self.tokenizer.mask_token
        sep_token = self.tokenizer.sep_token
        neighbor_token = self.neighbor_token

        head, rel, tail = self.entities[h], self.relations[r], self.entities[t]

        #因为存在某些entity没有desc,所以此处加一个判断
        if len(head)==3:
            h_name=head['name']
            h_desc=' '
        else:
            h_name, h_desc = head['name'], head['desc']
        r_name = rel['name']
        # 对于尾实体也一样,因为存在某些entity没有desc,所以此处加一个判断
        if len(tail) == 3:
            t_name = tail['name']
            t_desc = ' '
        else:
            t_name, t_desc = tail['name'], tail['desc']
        sep1, sep2, sep3, sep4 = rel['sep1'], rel['sep2'], rel['sep3'], rel['sep4']

        # 1. prepare inputs for nbert
        if self.encode_text:
            if self.add_neighbors:
                text_head_prompt = ' '.join(
                    [sep1, mask_token, sep2, r_name, sep3, t_name, neighbor_token, sep4, t_desc])
                text_tail_prompt = ' '.join(
                    [sep1, h_name, neighbor_token, sep2, r_name, sep3, mask_token, sep4, h_desc])
            else:
                text_head_prompt = ' '.join([sep1, mask_token, sep2, r_name, sep3, t_name, sep4, t_desc])
                text_tail_prompt = ' '.join([sep1, h_name, sep2, r_name, sep3, mask_token, sep4, h_desc])
        else:
            text_head_prompt, text_tail_prompt = None, None

        #
        # 2. prepare inputs for nformer
        if self.encode_struc:
            #self.vocab[t]是尾实体t的编号, [mask]的编号是1 其中t是实体代号(t,r-1,[mask])
            struc_head_prompt = [self.vocab[t], self.vocab[f'{r}_reverse'], self.vocab[mask_token]]
            struc_tail_prompt = [self.vocab[h], self.vocab[r], self.vocab[mask_token]]
        else:
            struc_head_prompt, struc_tail_prompt = None, None
        # 3. get filters
        head_filters = list(self.entity_filter[t, r] - {head['token_id']})
        tail_filters = list(self.entity_filter[h, r] - {tail['token_id']})
        # 4. prepare examples
        head_example = {
            'data_triple': (t, r, h),
            'data_text': (tail["raw_name"], r_name, head['raw_name']),
            'text_prompt': text_head_prompt,
            'struc_prompt': struc_head_prompt, #相当于是(尾,关系逆的,[mask])的一个编号
            'neighbors_label': tail['token_id'],
            'label': head["token_id"],
            'filters': head_filters,
        }
        tail_example = {
            'data_triple': (h, r, t),
            'data_text': (head['raw_name'], r_name, tail['raw_name']),
            'text_prompt': text_tail_prompt,
            'struc_prompt': struc_tail_prompt,
            'neighbors_label': head['token_id'],
            'label': tail["token_id"],
            'filters': tail_filters,
        }

        return head_example, tail_example

    def create_pretrain_examples(self):
        examples = dict()
        for mode in ['train', 'dev', 'test']:
            data = list()
            for h in self.entities.keys():
                name = str(self.entities[h]['name'])
                desc = str(self.entities[h]['desc'])
                desc_tokens = desc.split()

                prompts = [f'The description of {self.tokenizer.mask_token} is that {desc}']
                for i in range(10):
                    begin = random.randint(0, len(desc_tokens))
                    end = min(begin + self.max_seq_length, len(desc_tokens))
                    new_desc = ' '.join(desc_tokens[begin: end])
                    prompts.append(f'The description of {self.tokenizer.mask_token} is that {new_desc}')
                for prompt in prompts:
                    data.append({'prompt': prompt, 'label': self.entities[h]['token_id']})
            examples[mode] = data
        return examples

    def text_batch_encoding(self, inputs):
        encoded_data = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_seq_length)
        input_ids = torch.tensor(encoded_data['input_ids'])
        token_type_ids = torch.tensor(encoded_data['token_type_ids'])
        attention_mask = torch.tensor(encoded_data['attention_mask'])
        mask_pos = torch.nonzero(torch.eq(input_ids, self.tokenizer.mask_token_id))

        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
                'mask_pos': mask_pos}

    def struc_batch_encoding(self, inputs):
        input_ids = torch.tensor(inputs)
        return {'input_ids': input_ids}

    def collate_fn(self, batch_data):
        if self.task == 'pretrain':
            return self.collate_fn_for_pretrain(batch_data)

        # metadata
        data_triple = [data_dit['data_triple'] for data_dit in batch_data]  # [(h, r, t), ...]
        data_text = [data_dit['data_text'] for data_dit in batch_data]  # [(text, text, text), ...]

        text_prompts = [data_dit['text_prompt'] for data_dit in batch_data]  # [string, ...]
        text_data = self.text_batch_encoding(text_prompts) if self.encode_text else None
        struc_prompts = [copy.deepcopy(data_dit['struc_prompt']) for data_dit in batch_data]  # [string, ...]
        struc_data = self.struc_batch_encoding(struc_prompts) if self.encode_struc else None

        #sx 判断是否添加邻居信息
        if self.add_neighbors:
            batch_text_neighbors = [[] for _ in range(self.neighbor_num)]
            batch_struc_neighbors = [[] for _ in range(self.neighbor_num)]
            for ent, _, _ in data_triple:
                text_neighbors, struc_neighbors = self.neighbors[ent]['text_prompt'], self.neighbors[ent]['struc_prompt']
                idxs = list(range(len(text_neighbors)))
                if len(idxs) >= self.neighbor_num:
                    idxs = random.sample(idxs, self.neighbor_num)
                else:
                    tmp_idxs = []
                    for _ in range(self.neighbor_num - len(idxs)):
                        tmp_idxs.append(random.sample(idxs, 1)[0])
                    idxs = tmp_idxs + idxs
                assert len(idxs) == self.neighbor_num
                for i, idx in enumerate(idxs):
                    batch_text_neighbors[i].append(text_neighbors[idx])
                    batch_struc_neighbors[i].append(struc_neighbors[idx])
            # neighbor_num * batch_size
            text_neighbors = [self.text_batch_encoding(batch_text_neighbors[i]) for i in range(self.neighbor_num)] \
                if self.encode_text else None
            struc_neighbors = [self.struc_batch_encoding(batch_struc_neighbors[i]) for i in range(self.neighbor_num)] \
                if self.encode_struc else None
        else:
            text_neighbors, struc_neighbors = None, None

        neighbors_labels = torch.tensor([data_dit['neighbors_label']for data_dit in batch_data]) \
            if self.add_neighbors else None
        labels = torch.tensor([data_dit['label'] for data_dit in batch_data])
        filters = torch.tensor([[i, j] for i, data_dit in enumerate(batch_data) for j in data_dit['filters']])

        return {
            'data': data_triple,
            'data_text': data_text,
            'text_data': text_data,
            'text_neighbors': text_neighbors,
            'struc_data': struc_data,
            'struc_neighbors': struc_neighbors,
            'labels': labels,
            'filters': filters,
            'neighbors_labels': neighbors_labels,
        }

    def collate_fn_for_pretrain(self, batch_data):
        assert self.task == 'pretrain'

        lm_prompts = [data_dit['prompt'] for data_dit in batch_data]  # [string, ...]
        lm_data = self.text_batch_encoding(lm_prompts)

        labels = torch.tensor([data_dit['label'] for data_dit in batch_data])

        return {'text_data': lm_data, 'labels': labels, 'filters': None}

    #返回训练数据
    def get_train_dataloader(self):
        dataloader = DataLoader(self.train_ds, collate_fn=self.collate_fn,
                                batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=True)
        return dataloader

    def get_dev_dataloader(self):
        dataloader = DataLoader(self.dev_ds, collate_fn=self.collate_fn,
                                batch_size=2 * self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=False)
        return dataloader

    def get_test_dataloader(self):
        dataloader = DataLoader(self.test_ds, collate_fn=self.collate_fn,
                                batch_size=2 * self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=False)
        return dataloader

    def get_tokenizer(self):
        return self.tokenizer

def get_args():
    parser = argparse.ArgumentParser()
    # 1. about training
    parser.add_argument('--task', type=str, default='train', help='train | validate')
    # parser.add_argument('--model_path', type=str, default='', help='load saved model for validate') #加载已经保存的模型
    # parser.add_argument('--epoch', type=int, default=600, help='epoch')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size') #batch size
    parser.add_argument('--device', type=str, default='cpu', help='select a gpu like cuda:0')
    parser.add_argument('--dataset', type=str, default='UMLS-PubMed', help='select a dataset: fb15k-237 or wn18rr')
    # about neighbors
    parser.add_argument('--extra_encoder', action='store_true', default=False)
    parser.add_argument('--add_neighbors', action='store_true', default=True)
    parser.add_argument('--neighbor_num', type=int, default=4)  #
    parser.add_argument('--neighbor_token', type=str, default='[Neighbor]')
    parser.add_argument('--no_relation_token', type=str, default='[R_None]')
    # about struc encoder
    parser.add_argument('--kge_lr', type=float, default=2e-4)
    parser.add_argument('--kge_label_smoothing', type=float, default=0.8)
    # struc encoder config的参数
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=2)
    parser.add_argument('--input_dropout_prob', type=float, default=0.7, help='dropout before encoder')
    parser.add_argument('--context_dropout_prob', type=float, default=0.1, help='dropout for embeddings from neighbors')
    parser.add_argument('--attention_dropout_prob', type=float, default=0.1)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=768)   #为了和语义拼接,此处的参数需要手动调整,后续改成自动的.
    parser.add_argument('--intermediate_size', type=int, default=2048)
    parser.add_argument('--residual_dropout_prob', type=float, default=0.)
    parser.add_argument('--initializer_range', type=float, default=0.02)
    # 2. some unimportant parameters, only need to change when your server/pc changes, I do not change these
    parser.add_argument('--num_workers', type=int, default=32, help='num workers for Dataloader')
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory')
    # 3. convert to dict
    args = parser.parse_args()
    args = vars(args)

    # add some paths: tokenzier text_encoder data_dir output_dir
    root_path = os.path.dirname(__file__)
    print('root_path',root_path)
    # 1. tokenizer path
    args['tokenizer_path'] = "/home/h3c/data/xinSong/xs2/prompt-tuning/SapBERT-from-PubMedBERT-fulltext"


    # 2. saved model_path
    if args['task'] == 'validate':
        args['model_path'] = os.path.join(root_path, args['model_path'])
    # 3. data path
    args['data_path'] = os.path.join(root_path, 'dataset', args['dataset'])
    # 4. output path
    timestamp = strftime('%Y%m%d_%H%M%S', localtime())
    output_dir = os.path.join(root_path, 'output', args['dataset'], 'N-Former', timestamp)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    args['output_path'] = output_dir

    # save hyper params
    with open(os.path.join(args['output_path'], 'args.txt'), 'w') as f:
        json.dump(args, f, indent=4, ensure_ascii=False)

    # set random seed
    seed = 2022
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return args

def process_sx(all_data):
    keys = all_data['data']
    struc_datas = all_data['struc_data']['input_ids']
    struc_neighbors = all_data['struc_neighbors']
    neighbors_labels = all_data['neighbors_labels']
    new_data = {}
    for idx,key in enumerate(keys):
        key_str = "\t".join(key)
        struc_data = struc_datas[idx] #
        struc_neighbor = []
        for neigh in struc_neighbors:
            struc_neighbor.append(neigh['input_ids'][idx])
        neighbors_label = neighbors_labels[idx]
        new_data[key_str] = {
            'struc_data':struc_data,
            'struc_neighbor':struc_neighbor,
            'neighbors_label':neighbors_label
        }
    print("....")
    return new_data



if __name__ == '__main__':
    config = get_args()
    print('config', config)
    tokenizer_path = config['tokenizer_path']
    print(f'Loading Tokenizer from {tokenizer_path}')
    # sx 首先从网上下载最初的bert模型所需要的tokenizer
    # tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=False)
    tokenizer = BertTokenizer.from_pretrained("/home/h3c/data/xinSong/xs2/prompt-tuning/SapBERT-from-PubMedBERT-fulltext", do_basic_tokenize=False)

    data_module = KGCDataModule(config, tokenizer, encode_struc=True)  # 在dataset.py文件中
    tmp1 = data_module.collate_fn(data_module.train_stru)
    ttt = process_sx(tmp1)
    tmp2 = data_module.collate_fn(data_module.dev_stru)
    tmp3 = data_module.collate_fn(data_module.test_stru)
    print(data_module.train_stru)
    print(data_module.dev_stru)
    print(data_module.test_stru)


