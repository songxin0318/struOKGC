import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join
import torch.nn.functional as F

import re
import numpy as np
from transformers import AutoTokenizer
from p_tuning.models import get_embedding_layer, create_model
from data_utils.vocab import get_vocab_by_strategy, token_wrapper
from p_tuning.prompt_encoder import *
from struc_encoder.knowformer import Knowformer

class PTuneForLAMA(torch.nn.Module):

    def __init__(self, args, device, template, tokenizer_src, relation_num,config_str):
        super().__init__()
        self.args = args
        self.device = device      #  前面传的参数是'cuda:0'
        # self.device = "cpu"
        self.relation_num = relation_num
        self.tokenizer = tokenizer_src
        self.template = template

        self.model = create_model(self.args)
        self.model = self.model.to(self.device)   #修改

        for param in self.model.parameters():   #模型参数
            param.requires_grad = self.args.use_lm_finetune
        self.embeddings = get_embedding_layer(self.args, self.model)   #调用了models中的函数,实则是PLM的get_input_embeddings函数


        # set allowed vocab set  设置词汇vocab
        self.vocab = self.tokenizer.get_vocab()
        self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.args, self.tokenizer))

        # load prompt encoder     加载prompt编码器
        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.pseudo_token]
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
        self.spell_length = sum(self.template)

        #prompt编码器,在prompt_encoder.py文件中
        self.prompt_encoder = KEPromptEncoder(self.template, self.hidden_size, self.tokenizer, self.device, args, self.relation_num)
        self.prompt_encoder = self.prompt_encoder.to(self.device)     # 修改

        #for neighbors, we get the relative stru_embedding for soft prompt
        self.use_extra_encoder=False
        self.bert_encoder = Knowformer(config_str).to(self.device)


    #输入embedding
    # query 是三元组  rs 是标签
    def embed_input(self, queries, rs_tensor):

        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        replace_embeds = self.prompt_encoder(rs_tensor)
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[bidx, i, :]
        return raw_embeds

    def forward_classification(self, texts, rs, labels, batch_neighbors, return_candidates=False):
        #1  global_id接收batchdata[3]中存放的三元组数据. ['t\r\t','',...]
        #2 其中batch_neighbors是个字典,存放三元组的结果信息,用于结构编码器的输入
        #batch_neighbors['struc_data']
        # batch_neighbors['struc_neighbor']
        # batch_neighbors['neighbors_label']

        if self.args.model_name == 'luke':
            return self.forward_classification_luke(texts, rs, labels, return_candidates)   #调用自定义的分类函数
        bz = len(texts)   # 其实就是batch-size
        # print('modeling-texts-length:',bz)

        # construct query ids
        prompt_tokens = [self.pseudo_token_id]  # prompt_tokens  [50265]
        #调用prompt_encoder.py里面的get.enquery() 函数,获得的是一个batch 的
        queries = [torch.LongTensor(self.prompt_encoder.get_query(texts[i], rs[i], prompt_tokens)).squeeze(0) for i in range(bz)]  #对definition拼接
        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)  # 对拼接之后的query, 字节的填充[batch,$] for wiki [16,25]
        #此时的querises 是16*32(batch*toeken_num)

        # construct label ids
        attention_mask = queries != self.pad_token_id
        #添加一个维度之后,需要修改attention_mask
        my_pad = torch.ones(attention_mask.shape[0],1)
        my_ppad = my_pad == 1
        my_ppad = my_ppad.to(self.device)
        # attention_mask.to(self.device)
        attention_mask= torch.cat([attention_mask,my_ppad],dim=1)
        #
        rs_tensor = torch.LongTensor(rs).to(self.device)

        # get embedded input
        inputs_embeds = self.embed_input(queries, rs_tensor)  #
        # print("$$$$$$$inputs_embeds-origin")  # [16,25(pad 之后的字节长度),768]  #bert的embedding 维度


        # global_id接收batchdata[3]中存放的三元组数据. ['t\r\t','',...]

        for i in range(bz):
            if batch_neighbors['struc_data'][i] is None:  #如果邻居为空，
                batch_neighbors['struc_data'][i]=torch.tensor( [1,1,1] )  #结构为none,用自学习的prompt_tokens
        input_ids= torch.stack(batch_neighbors['struc_data']) # batch_size*3  调整输入格式相同.

        context_input_ids=[]#[[],[],[],[]]
        neigh_num=2  # 邻居
        for n_num in range(neigh_num): #遍历邻居

            context_input_ids_n=[]
            for j in range(bz):
                if batch_neighbors['struc_neighbor'][j] is None:
                    context_input_ids_n.append([1,1,1])
                else:
                    context_input_ids_n.append(batch_neighbors['struc_neighbor'][j][n_num].numpy().tolist())
            context_input_ids.append(torch.tensor(context_input_ids_n))
        # print('context_input_ids 2 ---------')
        output_struc = self.bert_encoder(input_ids.to(self.device), context_input_ids, self.use_extra_encoder)   #_neighbors
        struc_emb=output_struc['with_neighbors'] #
            #struc_emb的size 是 16*256 其中16是batch size
        # print('$$$$$$$ output_struc: ')


        #self.model  调用的是models中的
        # output = self.model(inputs_embeds=inputs_embeds.to(self.device),
        #                     attention_mask=attention_mask.to(self.device).bool(),
        #                     labels=torch.LongTensor(labels).to(self.device))   #labels torch.LongTensor(labels)
        # 对三元组的 语义信息 inputs_embeds cuda 0 和 局部结构 out_struc 信息进行结合;
        mix_triple_struc=torch.cat([inputs_embeds,struc_emb.unsqueeze(1).to(self.device)],1)

        output = self.model(inputs_embeds=mix_triple_struc.to(self.device),
                            attention_mask=attention_mask.to(self.device).bool(),
                            labels=torch.LongTensor(labels).to(self.device))
         # self.tok
        # print('****out-put******')
        loss, logits = output.loss, output.logits

        # print('loss',logits)
        #torch.argmax()函数
        #dim=0时，返回每一列最大值的索引;
        # dim = 1时，返回每一行最大值的索引
        acc = torch.sum(torch.argmax(logits, dim=-1) == torch.LongTensor(labels).to(self.device))  #列最大值准确率

        # return loss, float(acc) / bz, (labels.tolist(), torch.argmax(logits, dim=-1).tolist(), logits)
        return loss, float(acc) / bz, (labels, torch.argmax(logits, dim=-1).tolist(), logits)
