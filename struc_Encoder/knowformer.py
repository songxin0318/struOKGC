import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .struformer_encoder import Embeddings, Encoder, truncated_normal_init, norm_layer_init
import time

'''
定义knowformet类
'''
class Knowformer(nn.Module):
    def __init__(self, config):
        super(Knowformer, self).__init__()
        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']  #2
        self._n_head = config['num_attention_heads']  #2
        self._input_dropout_prob = config['input_dropout_prob']  #0.7
        self._attention_dropout_prob = config['attention_dropout_prob'] #0.1
        self._hidden_dropout_prob = config['hidden_dropout_prob']  #0.1
        self._residual_dropout_prob = config['residual_dropout_prob'] #0.0
        self._context_dropout_prob = config['context_dropout_prob'] #0.1
        self._initializer_range = config['initializer_range'] #0.02
        self._intermediate_size = config['intermediate_size']  #2048

        self._voc_size = config['vocab_size'] #sx-important  #sapbert 9757
        self._n_relation = config['num_relations']

        #传入的参数是self._emb_size
        self.ele_embedding = Embeddings(self._emb_size, self._voc_size, self._initializer_range).to("cuda:0")

        self.triple_encoder = Encoder(config).to("cuda:0")   #三元组编码器
        self.context_encoder = Encoder(config).to("cuda:0")   #上下文信息编码器

        # 对输入的准备(包括两部分,三元组输入准备 和 上下文信息输入准备)
        self.input_dropout_layer = nn.Dropout(p=self._input_dropout_prob)
        self.context_dropout_layer = nn.Dropout(p=self._context_dropout_prob)

    def __forward_triples(self, triple_ids, context_emb=None, encoder_type="triple"):
        # convert token id to embedding 把三元组的id转化为embedding
        emb_out = self.ele_embedding(triple_ids)  # (batch_size, 3, embed_size)    输入张量大小

        # merge context_emb into emb_out
        if context_emb is not None:  #如果三元组的上下文信息 context_emb 不为空,则 首先 经过输入准备,然后 与三与组自身算数平均
            context_emb = self.context_dropout_layer(context_emb)
            emb_out[:, 0, :] = (emb_out[:, 0, :] + context_emb) / 2

        emb_out = self.input_dropout_layer(emb_out)
        encoder = self.triple_encoder if encoder_type == "triple" else self.context_encoder
        emb_out = encoder(emb_out, mask=None)   # (batch_size, 3, embed_size)
        # 经过__forward_triples之后,return emb_out,该张量是融合了三元组和上下文信息的向量
        return emb_out

    def __process_mask_feat(self, mask_feat):
        return torch.matmul(mask_feat, self.ele_embedding.lut.weight.transpose(0, 1))

    def forward(self, src_ids, window_ids=None, double_encoder=False):
        # src_ids: (batch_size, seq_size, 1)
        # window_ids: (batch_size, seq_size) * neighbor_num

        # 1. do not use embeddings from neighbors   不使用邻居信息的embedding
        seq_emb_out = self.__forward_triples(src_ids, context_emb=None)  #调用前面的__forward_triples获得三元组的embedding
        mask_emb = seq_emb_out[:, 2, :]  # (batch_size, embed_size)
        logits_from_triplets = self.__process_mask_feat(mask_emb)  # (batch_size, vocab_size)

        if window_ids is None:
            return {'without_neighbors': logits_from_triplets, 'with_neighbors': None, 'neighbors': None}

        # 2. encode neighboring triplets  使用邻居信息  邻居信息存储在参数window_ids中
        logits_from_neighbors = []
        embeds_from_neighbors = []
        for i in range(len(window_ids)):
            if double_encoder:
                seq_emb_out = self.__forward_triples(window_ids[i].to("cuda:0"), context_emb=None, encoder_type='context')
            else:
                seq_emb_out = self.__forward_triples(window_ids[i].to("cuda:0"), context_emb=None, encoder_type='triple')
            mask_emb = seq_emb_out[:, 2, :]  #最后一维,就是尾实体的embedding
            logits = self.__process_mask_feat(mask_emb)

            embeds_from_neighbors.append(mask_emb)
            logits_from_neighbors.append(logits)
        # get embeddings from neighboring triplets by averaging
        context_embeds = torch.stack(embeds_from_neighbors, dim=0)  # (neighbor_num, batch_size, 256)
        context_embeds = torch.mean(context_embeds, dim=0)

        # 3. leverage both the triplet and neighboring triplets
        seq_emb_out = self.__forward_triples(src_ids, context_emb=context_embeds)  #context_embeds 16*256
        mask_embed = seq_emb_out[:, 2, :] #最后一维,就是尾实体的embedding

        # logits_from_both = self.__process_mask_feat(mask_embed)
        return {
            'with_neighbors': mask_embed
        }

