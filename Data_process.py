# -*- coding:utf-8 -*-
from collections import Counter
import pickle
import numpy as np
import tensorflow.contrib.keras as kr

# 将tag转换成数字
tag2label = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}

def read_data(filename):
    '''
    :param filename:
    :return: 将文字与标签分开，存入列表
    '''
    content, label, sentences, tag = [], [], [], []
    with open(filename, encoding='utf-8') as file:
        lines = file.readlines()
    for eachline in lines:
        if eachline != '\n':
            [char, tag_] = eachline.strip().split()
            sentences.append(char)
            tag.append(tag_)
        else:
            content.append(sentences)
            label.append(tag)
            sentences, tag = [], []
    return content, label

#data = read_data('./data/train_data')

def build_vocab(filenames, vocab_size = 5000):
    '''
    :param filenames: 所有数据
    :param vocab_size: 取频率最高的5000字
    :return:
    '''
    wordlist = []
    word = {}
    word['<PAD>'] = 0
    j = 1
    for filename in filenames:
        content, _ = read_data(filename)
        for sen_ in content:
            wordlist.extend(sen_)
    counter = Counter(wordlist)
    count_pari = counter.most_common(vocab_size)
    word_, _ = list(zip(*count_pari))
    for key in word_:
        if key.isdigit():
            key = '<NUM>'
        if key not in word:
            word[key] = j
        j += 1

    word['<UNK>'] = j
    with open('./data/word2id.pkl', 'wb') as fw: #将建立的字典 保存
        pickle.dump(word, fw)
    return word
#filenames = ['./data/train_data','./data/test_data']
#build_vocab(filenames)
#with open('./data/word2id.pkl', 'rb') as fr:
    #word2id = pickle.load(fr)
#print(word2id)
#print(len(word2id))

def sequence2id(filename):
    '''
    :param filename:
    :return: 将文字与标签，转换为数字
    '''
    content2id, label2id = [], []
    content, label = read_data(filename)
    with open('./data/word2id.pkl', 'rb') as fr:
        word = pickle.load(fr)
    for i in range(len(label)):
        label2id.append([tag2label[x] for x in label[i]])
    for j in range(len(content)):
        w = []
        for key in content[j]:
            if key.isdigit():
                key = '<NUM>'
            elif key not in word:
                key = '<UNK>'
            w.append(word[key])
        content2id.append(w)
    return content2id, label2id

#content2id, label2id = sequence2id('./data/train_data')
#print(len(content2id[1]))
#print(len(label2id[1]))


def batch_iter(x, y, batch_size=64):
    '''
    :param x: content2id
    :param y: label2id
    :param batch_size: 每次进入模型的句子数量
    :return:
    '''
    data_len = len(x)
    x = np.array(x)
    y = np.array(y)
    num_batch = int((data_len - 1) / batch_size) + 1 #计算一个epoch,需要多少次batch

    indices = np.random.permutation(data_len) #生成随机数列
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = batch_size * i
        end_id = min(batch_size*(i+1), data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def process_seq(x_batch):
    '''
    :param x_batch: 计算一个batch里面最长句子 长度n
    :param y_batch:动态RNN 保持同一个batch里句子长度一致即可，sequence为实际句子长度
    :return: 对所有句子进行padding,长度为n
    '''
    seq_len = []
    max_len = max(map(lambda x: len(x), x_batch))  # 计算一个batch中最长长度
    for i in range(len(x_batch)):
        seq_len.append(len(x_batch[i]))

    x_pad = kr.preprocessing.sequence.pad_sequences(x_batch, max_len, padding='post', truncating='post')
    #y_pad = kr.preprocessing.sequence.pad_sequences(y_batch, max_len, padding='post', truncating='post')

    return x_pad, seq_len


#batch_train = batch_iter(content2id, label2id, batch_size=64)
#for x_batch, y_batch in batch_train:
    #x_pad, y_pad, seq_len = process_seq(x_batch, y_batch)
    #print(x_pad[1])
    #print(len(x_pad[1]))
    #print(y_pad[1])
    #print(len(y_pad[1]))
    #print(seq_len[1])