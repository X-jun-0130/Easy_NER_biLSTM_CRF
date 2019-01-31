# -*- coding:utf-8 -*-
class Parameters(object):

    num_epochs = 12
    vocab_size = 4834
    embedding_size = 100
    batch_size = 64
    hidden_dim = 128
    learning_rate = 0.001
    clip = 5.0
    lr = 0.9
    keep_pro = 0.5
    num_tags = 7


    train_data = './data/train_data'
    test_data = './data/test_data'
    eva = './data/eva_data'