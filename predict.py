# -*- coding: utf-8 -*-
import os
import sys
import torch
import logging
import hydra
import models
from hydra import utils
from utils import load_pkl, load_csv
from serializer import Serializer
from preprocess import _serialize_sentence, _convert_tokens_into_index, _add_pos_seq, _handle_relation_data
import matplotlib.pyplot as plt




def _preprocess_data(data, cfg):
    serializer = Serializer(do_chinese_split=cfg.chinese_split)
    serial = serializer.serialize

    _serialize_sentence(data, serial, cfg)
    _convert_tokens_into_index(data, vocab)
    _add_pos_seq(data, cfg)
    
    return data, rels


def _get_predict_instance(cfg, line):
    sentence, head, tail = line.split(' ')
    instance = dict()
    instance['sentence'] = sentence.strip()
    instance['head'] = head.strip()
    instance['tail'] = tail.strip()
    cfg.replace_entity_with_type = False
    instance['head_type'] = 'None'
    instance['tail_type'] = 'None'
    return instance


# 自定义模型存储的路径
cw = os.path.abspath('.') + os.path.sep
fp = cw + 'cnn_epoch22.pth'
if os.path.exists(fp) == False:
    print('no .pth file exists in', os.path.abspath('.'))
    sys.exit()

@hydra.main(config_path='conf/config.yaml')
def main(cfg):

    cwd = utils.get_original_cwd()
    cfg.cwd = cwd
    cfg.pos_size = 2 * cfg.pos_limit + 2

    # get predict instance
    #instance = _get_predict_instance(cfg)
    #data = [instance]

    global vocab,rels
    vocab = load_pkl(os.path.join(cfg.cwd, cfg.out_path, 'vocab.pkl'), verbose=False)
    relation_data = load_csv(os.path.join(cfg.cwd, cfg.data_path, 'relation.csv'), verbose=False)
    rels = _handle_relation_data(relation_data)
    cfg.vocab_size = vocab.count

    # preprocess data
    #data, rels = _preprocess_data(data, cfg)

    # model
    __Model__ = {
        'cnn': models.PCNN,
        'rnn': models.BiLSTM,
        'transformer': models.Transformer,
        'gcn': models.GCN,
        'capsule': models.Capsule,
        'lm': models.LM,
    }

    # 最好在 cpu 上预测
    # cfg.use_gpu = False
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', cfg.gpu_id)
    else:
        device = torch.device('cpu')


    model = __Model__[cfg.model_name](cfg)

    model.load(fp, device=device)
    model.to(device)
    model.eval()



    file = open(cw + 'data' + os.path.sep + 'out.txt', encoding = "utf8")
    out = open(cw + 'data' + os.path.sep + 'predict.txt', 'w', encoding = "utf8")
    #last_line = ''
    for line in file:
        # get predict instance
        instance = _get_predict_instance(cfg, line)
        data = [instance]
        # preprocess data
        data, rels = _preprocess_data(data, cfg)


        x = dict()
        x['word'], x['lens'] = torch.tensor([data[0]['token2idx']]), torch.tensor([data[0]['seq_len']])
        if cfg.model_name != 'lm':
            x['head_pos'], x['tail_pos'] = torch.tensor([data[0]['head_pos']]), torch.tensor([data[0]['tail_pos']])
            if cfg.model_name == 'cnn':
                if cfg.use_pcnn:
                    x['pcnn_mask'] = torch.tensor([data[0]['entities_pos']])

        for key in x.keys():
            x[key] = x[key].to(device)

        with torch.no_grad():
            y_pred = model(x)
            y_pred = torch.softmax(y_pred, dim=-1)[0]
            prob = y_pred.max().item()
            prob_rel = list(rels.keys())[y_pred.argmax().item()]
            if prob_rel == 'unknown':
                continue
            if prob < 0.96:
                continue
            out.write(f"\"{data[0]['head']}\" is-a \"{data[0]['tail']}\" 的置信度为{prob:.2f}。\tsentence: {line.split(' ')[0]}\n")
            print(f"\"{data[0]['head']}\" is-a \"{data[0]['tail']}\" 的置信度为{prob:.2f}。\tsentence: {line.split(' ')[0]}")



if __name__ == '__main__':
    main()
