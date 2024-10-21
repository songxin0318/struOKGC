from torch.utils.data import DataLoader
from transformers import BertTokenizer,AutoTokenizer

from data_utils.dataset import *
from os.path import join, abspath, dirname

# import struc_encoder.KG_neighhboors
from struc_encoder.KG_neighhboors import get_args, KGCDataModule, process_sx


def sx_process1():
    config = get_args()
    print('config', config)
    # tokenizer_path = config['tokenizer_path']
    # print(f'Loading Tokenizer from {tokenizer_path}')
    # sx 首先从网上下载最初的bert模型所需要的tokenizer
    # tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=False)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_basic_tokenize=False)
    tokenizer = BertTokenizer.from_pretrained('/home/h3c/data/xinSong/xs2/prompt-tuning/SapBERT-from-PubMedBERT-fulltext')

    data_module = KGCDataModule(config, tokenizer, encode_struc=True)  # 在dataset.py文件中
    tokenizer=data_module.get_tokenizer()
    train_new_data = process_sx(data_module.collate_fn(data_module.train_stru)) #经历上面之后，token变成了，加上实体和关系的数值了。
    dev_new_data = process_sx(data_module.collate_fn(data_module.dev_stru))
    test_new_data = process_sx(data_module.collate_fn(data_module.test_stru))

    return tokenizer,train_new_data, dev_new_data, test_new_data, config


def collate_fn_sx(origin_data, new_data):
    # print("进入加工collate_fn_sx函数..")
    all_data = {}
    all_data['triple_text'] = []
    all_data['rs'] = []
    all_data['label'] = []
    all_data['triple_str'] = []
    all_data['struc_neighbor'] = []
    all_data['struc_data'] = []
    all_data['neighbors_label'] = []
    for data in origin_data:
        triple_text = data[0]
        rs = data[1]
        label = data[2]
        triple_str = data[3]
        all_data['triple_text'].append(triple_text)
        all_data['rs'].append(rs)
        all_data['label'].append(label)
        all_data['triple_str'].append(triple_str)
        if new_data.__contains__(triple_str):
            res = new_data[triple_str]
            all_data['struc_neighbor'].append(res['struc_neighbor'])
            all_data['struc_data'].append(res['struc_data'])
            all_data['neighbors_label'].append(res['neighbors_label'])
        else:
            all_data['struc_neighbor'].append(None)
            all_data['struc_data'].append(None)
            all_data['neighbors_label'].append(None)
    # print(",,,")

    return all_data

def collate_fn_sx_link(origin_data, new_data):
    # print("进入加工collate_fn_sx函数..")
    all_data = {}
    all_data['triple_text'] = []
    all_data['rs'] = []
    all_data['label'] = []
    all_data['triple_str'] = []
    all_data['struc_neighbor'] = []
    all_data['struc_data'] = []
    all_data['neighbors_label'] = []
    for data in origin_data:
        triple_text = data[0]
        rs = data[1]
        label = data[2]
        triple_str = data[3]
        all_data['triple_text'].append(triple_text)
        all_data['rs'].append(rs)
        all_data['label'].append(label)
        all_data['triple_str'].append(triple_str)
        if new_data.__contains__(triple_str):
            res = new_data[triple_str]
            all_data['struc_neighbor'].append(res['struc_neighbor'])
            all_data['struc_data'].append(res['struc_data'])
            all_data['neighbors_label'].append(res['neighbors_label'])
        else:
            all_data['struc_neighbor'].append(None)
            all_data['struc_data'].append(None)
            all_data['neighbors_label'].append(None)
    # print(",,,")

    return all_data
# def collate_fn(origin_data):
#     # origin_data shape batch
#     # #为了匹配邻居
#     # train_new_data, dev_new_data, test_new_data = sx_process1()
#     new_data, _, _ = sx_process1()
#     all_data = {}
#     all_data['triple_text'] = []
#     all_data['rs'] = []
#     all_data['label'] = []
#     all_data['triple_str'] = []
#     all_data['struc_neighbor'] = []
#     all_data['struc_data'] = []
#     all_data['neighbors_label'] = []
#     # for data in origin_data:
#     #     triple_text = data[0]
#     #     rs = data[1]
#     #     label = data[2]
#     #     triple_str = data[3]
#     #     all_data['triple_text'].append(triple_text)
#     #     all_data['rs'].append(rs)
#     #     all_data['label'].append(label)
#     #     all_data['triple_str'].append(triple_str)
#     #     if new_data.__contains__(triple_str):
#     #         res = new_data[triple_str]
#     #         all_data['struc_neighbor'].append(res['struc_neighbor'])
#     #         all_data['struc_data'].append(res['struc_data'])
#     #         all_data['neighbors_label'].append(res['neighbors_label'])
#     #     else:
#     #         all_data['struc_neighbor'].append(None)
#     #         all_data['struc_data'].append(None)
#     #         all_data['neighbors_label'].append(None)
#     print(",,,")
#     for unit in origin_data:
#         print(unit)
#     return all_data
#
# def collate_fn_sx_1(origin_data):
#     # #为了匹配邻居
#     # train_new_data, dev_new_data, test_new_data = sx_process1()
#     new_data, _, _ = sx_process1()
#     all_data = {}
#     all_data['triple_text'] = []
#     all_data['rs'] = []
#     all_data['label'] = []
#     all_data['triple_str'] = []
#     all_data['struc_neighbor'] = []
#     all_data['struc_data'] = []
#     all_data['neighbors_label'] = []
#     for data in origin_data:
#         triple_text = data[0]
#         rs = data[1]
#         label = data[2]
#         triple_str = data[3]
#         all_data['triple_text'].append(triple_text)
#         all_data['rs'].append(rs)
#         all_data['label'].append(label)
#         all_data['triple_str'].append(triple_str)
#         if new_data.__contains__(triple_str):
#             res = new_data[triple_str]
#             all_data['struc_neighbor'].append(res['struc_neighbor'])
#             all_data['struc_data'].append(res['struc_data'])
#             all_data['neighbors_label'].append(res['neighbors_label'])
#         else:
#             all_data['struc_neighbor'].append(None)
#             all_data['struc_data'].append(None)
#             all_data['neighbors_label'].append(None)
#     # print(",,,")
#     return all_data

def get_dataloader(args, tokenizer):
    basic_data = BasicDataWiki(args, tokenizer)  # dataset包

    # 基于KGE的负三元组,通过 将头部或尾部实体替换 为KGE模型 认为具有高概率 持有的另一个实体  而生成的。
    neg_file_kge = join(args.data_dir, f'train_neg_kge_{args.keg_neg}.txt')
    if args.random_neg_ratio == 1.0:
        neg_file_kge = None

    train_set = KEDatasetWiki(
        join(args.data_dir, 'train.txt'),  # 事实三元组
        join(args.data_dir, 'train_neg_rand.txt'),  # 随机负三元组___用实体集合E中其他实体，随机替换T中三元组的头或者尾实体来获得的
        basic_data,
        neg_file_kge=neg_file_kge,
        pos_K=args.pos_K,
        neg_K=args.neg_K,
        random_neg_ratio=args.random_neg_ratio
    )
    test_set = KEDatasetWiki(
        join(args.data_dir, 'test_pos.txt'),
        join(args.data_dir, 'test_neg.txt'),
        basic_data
    )
    dev_set = KEDatasetWiki(
        join(args.data_dir, 'valid_pos.txt'),
        join(args.data_dir, 'valid_neg.txt'),
        basic_data
    )

    if args.test_open:
        o_test_set = KEDatasetWiki(
            join(args.data_dir, 'o_test_pos.txt'),
            join(args.data_dir, 'o_test_neg.txt'),
            basic_data
        )
    if args.link_prediction:
        link_dataset_tail = KEDatasetWikiInfer(
            join(args.data_dir, 'link_prediction_tail.txt'),
            basic_data,
            args.recall_k
        )
        link_dataset_head = KEDatasetWikiInfer(
            join(args.data_dir, 'link_prediction_head.txt'),
            basic_data,
            args.recall_k
        )

    print('sxsxsxsxsx',basic_data)

    # #为了匹配邻居
    tokenize_stru, train_new_data, dev_new_data, test_new_data, config = sx_process1()
    # #构造新的数据
    # train_pre = collate_fn_sx(train_set,train_new_data)

    # train_loader = DataLoader(train_pre, batch_size=args.batch_size, shuffle=True) #sx 新的train_set,换成train_pre
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,collate_fn=lambda x:collate_fn_sx(x,train_new_data), shuffle=True)
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=lambda x: collate_, shuffle=True)
    # dev_loader = DataLoader(dev_set, batch_size=args.batch_size)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,collate_fn=lambda x:collate_fn_sx(x,dev_new_data))
    # test_loader = DataLoader(test_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,collate_fn=lambda x:collate_fn_sx(x,test_new_data))

    ch_test_loader, oh_test_loader = None, None

    if args.test_open:
        o_test_loader = DataLoader(o_test_set, batch_size=args.batch_size)
    else:
        o_test_loader = None

    if args.link_prediction:
        #字典拼接

        train_new_data.update(dev_new_data)
        train_new_data.update(test_new_data)
        link_loader_tail = DataLoader(link_dataset_tail, batch_size=args.batch_size,collate_fn=lambda x:collate_fn_sx_link(x,train_new_data))
        link_loader_head = DataLoader(link_dataset_head, batch_size=args.batch_size,collate_fn=lambda x:collate_fn_sx_link(x,train_new_data))
        # link_loader_tail = DataLoader(link_dataset_tail, batch_size=args.batch_size)
        # link_loader_head = DataLoader(link_dataset_head, batch_size=args.batch_size)
    else:
        link_loader_tail = None
        link_loader_head = None
        link_dataset_tail = None
        link_dataset_head = None
    print('sx  get_dataloader函数', '文件加载完毕')
    return train_loader, dev_loader, test_loader, ch_test_loader, oh_test_loader, o_test_loader, link_loader_head, link_loader_tail, len(
        basic_data.relation2idx), link_dataset_head, link_dataset_tail, config,tokenize_stru,
