# !/usr/bin/python2.7
# -*- coding: utf-8 -*-
 
##############################################################
# 
# Copyright (c) 2018 USTC, Inc. All Rights Reserved
# 
##############################################################
# 
# File:    SRULM.py
# Author:  roee
# Date:    2018/03/05 19:30:26
# Brief:
# 
# 
##############################################################

import tensorflow as tf
import numpy as np
import os
import math
import collections
import pickle
import sys
import random
import time
from tensorflow.python.ops.rnn_cell_impl import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 0 or 1 or 2 ..


def file2list(input_file):
    output_list = []
    for line in input_file:
        sentence = line.replace('\n', ' <eos> ').split()
        output_list.extend(sentence)
    return output_list

def build_dataset(words, vocabulary_pro):
    vocab = collections.Counter(words)
    if '<unk>' in vocab:
        count = []
    else:
        count = [['<unk>', 0]]
    size = len(vocab)
    l = vocab.most_common(int(size * vocabulary_pro))
    count.extend([[l[i][0], l[i][1]] for i in range(len(l))])
    word2id = {}
    for word, _ in count:
        word2id[word] = len(word2id)
    data = []
    unk_count = 0
    for word in words:
        if word in word2id:
            if word == '<unk>':
                unk_count += 1
            index = word2id[word]
            data.append(index)
        else:
            index = word2id['<unk>']
            data.append(index)
            unk_count += 1
    for i in range(len(count)):
        if count[i][0] == '<unk>':
            count[i][1] = unk_count
            break
    id2word = dict(zip(word2id.values(), word2id.keys()))
    return data, count, word2id, id2word


def LM(embeddings, X):
    x = tf.reshape(tf.nn.embedding_lookup(embeddings, 
                tf.reshape(X, [BATCH_SIZE * WIN_NUM])), [BATCH_SIZE, WIN_NUM, EMBED_SIZE])
    x = tf.nn.dropout(x, keep_prob)
    def cell_creator():
        return BasicLSTMCell(OUT_SIZE)
    cell = tf.contrib.rnn.MultiRNNCell([cell_creator() for _ in range(2)], state_is_tuple=True)
    h = cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    o, state = tf.nn.dynamic_rnn(cell=cell, inputs=x, initial_state=h, time_major=False)
    return o
   

if __name__ == "__main__":
    
    #if len(sys.argv) != 3:
    #    print 'python '+ sys.argv[0] +' corpus/ptb/ vector/ptb/rnn'
    #    sys.exit()
    #===Paras============================================
    para = {}
    para['init_scale'] = 0.1
    para['batch_size'] = 64
    para['win_num'] = 20
    para['embed_size'] = 50
    para['hidden_size'] = 50
    para['max_max_epoch'] = 13
    para['max_epoch'] = 4
    para['keep_prob'] = 1.0
    para['lr_decay'] = 0.5
    para['vocab_pro'] = 1.0
    para['learning_rate'] = 1.0
    para['max_grad_norm'] = 5
    BATCH_SIZE = para['batch_size']   
    WIN_NUM = para['win_num']
    EMBED_SIZE = para['embed_size']
    OUT_SIZE = para['hidden_size']
    EPOCH_SIZE = para['max_max_epoch']
    VOCAB_PRO = para['vocab_pro']
    init_scale = para['init_scale']
    max_epoch = para['max_epoch']
    keep_prob = para['keep_prob']
    lr_decay = para['lr_decay'] 
    learning_rate = para['learning_rate']
    max_grad_norm = para['max_grad_norm']

    for p in para:
        print p + ':\t' + str(para[p])
    #===Read_corpus============================================
    train_file = open('corpus/ptb/'+'train.txt', 'r')
    valid_file = open('corpus/ptb/'+'valid.txt', 'r')
    test_file = open('corpus/ptb/'+'test.txt', 'r')
    train = file2list(train_file)
    data, count, word2id, id2word = build_dataset(train, VOCAB_PRO) 
    VOCAB_SIZE = len(count)
    print 'All vocabulary size: ', VOCAB_SIZE
    DATA_LEN = len(data)
    BATCH_LEN = DATA_LEN // BATCH_SIZE
    EPOCH_LEN = (BATCH_LEN - 1) // (WIN_NUM)
    dat = np.zeros([BATCH_SIZE, BATCH_LEN], dtype=np.int32)
    for i in range(BATCH_SIZE):
        dat[i] = data[BATCH_LEN * i:BATCH_LEN * (i+1)]
    num_steps = WIN_NUM
    train_data = []
    for i in range(EPOCH_LEN):
        x = dat[:, i*num_steps:(i+1)*num_steps]
        y = dat[:, i*num_steps+1:(i+1)*num_steps+1]
        train_data.append([x, y])
    test = []
    for line in test_file:
        sentence = line.replace('\n', ' <eos> ').split()
        sen = []
        for word in sentence:
            if word in word2id:
                sen.append(word2id[word])
            else:
                sen.append(word2id['<unk>'])
        test.extend(sen)
    data_len = len(test)
    batch_len = data_len // BATCH_SIZE
    epoch_len = (batch_len - 1) // num_steps
    dat = np.zeros([BATCH_SIZE, batch_len], dtype=np.int32)
    for i in range(BATCH_SIZE):
        dat[i] = test[batch_len * i:batch_len * (i+1)]
    test_data = []
    for i in range(epoch_len):
        x = dat[:, i*num_steps:(i+1)*num_steps]
        y = dat[:, i*num_steps+1:(i+1)*num_steps+1]
        test_data.append([x, y])
    valid = []
    for line in valid_file:
        sentence = line.replace('\n', ' <eos> ').split()
        sen = []
        for word in sentence:
            if word in word2id:
                sen.append(word2id[word])
            else:
                sen.append(word2id['<unk>'])
        valid.extend(sen)
    data_len = len(valid)
    batch_len = data_len // BATCH_SIZE
    epoch_len = (batch_len - 1) // num_steps
    dat = np.zeros([BATCH_SIZE, batch_len], dtype=np.int32)
    for i in range(BATCH_SIZE):
        dat[i] = valid[batch_len * i:batch_len * (i+1)]
    valid_data = []
    for i in range(epoch_len):
        x = dat[:, i*num_steps:(i+1)*num_steps]
        y = dat[:, i*num_steps+1:(i+1)*num_steps+1]
        valid_data.append([x, y])
    TRAIN_STEPS = len(data) * EPOCH_SIZE / BATCH_SIZE
    #===Train============================================
    X = tf.placeholder(tf.int32, shape=[BATCH_SIZE, WIN_NUM])
    Y = tf.placeholder(tf.int32, shape=[BATCH_SIZE, WIN_NUM])
    embeddings = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -init_scale, init_scale))
    output = tf.reshape(LM(embeddings, X), [-1, OUT_SIZE])
    loss_weights =tf.Variable(tf.random_uniform([OUT_SIZE, VOCAB_SIZE], -init_scale, init_scale))
    loss_biases = tf.Variable(tf.random_uniform([VOCAB_SIZE], -init_scale, init_scale))
    logits = tf.matmul(output, loss_weights) + loss_biases
    cost = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], 
            [tf.reshape(Y, [-1])], [tf.ones([BATCH_SIZE*WIN_NUM], dtype=tf.float32)])
    logppl = tf.reduce_sum(cost) / BATCH_SIZE / WIN_NUM 
    params = tf.trainable_variables()
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    gradients = tf.gradients(cost, params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_grad_norm)
    train_op = opt.apply_gradients(zip(clipped_gradients,
                params),global_step=tf.contrib.framework.get_or_create_global_step())


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    print 'Epoch size: ', EPOCH_SIZE
    print 'Iterations: ', TRAIN_STEPS
    for i in range(TRAIN_STEPS+1):
        if (i%BATCH_LEN) == 0:
            if i//BATCH_LEN >= max_epoch:
                learning_rate *= lr_decay
            ppls = 0
            for batch, label in valid_data:
                ppl = sess.run(logppl, feed_dict={X: batch, Y: label})
                ppls += ppl
            valid_ppl = np.exp(ppls / len(valid_data))
            ppls = 0
            for batch, label in test_data:
                ppl = sess.run(logppl, feed_dict={X: batch, Y: label})
                ppls += ppl
            test_ppl = np.exp(ppls / len(test_data))
            batch, label = train_data[i%EPOCH_LEN][0], train_data[i%EPOCH_LEN][1]
            _, ppl = sess.run([train_op, logppl], feed_dict={X: batch, Y: label})
            train_ppl = np.exp(ppl)
            TimeTuple=time.localtime(time.time())
            fmt='%Y-%m-%d %a %H:%M:%S'
            tt=time.strftime(fmt,TimeTuple)
            print '[%s]\t%d\t%.3f\t%.3f\t%.3f\t%.3f' % (tt, i//BATCH_LEN, learning_rate, 
                train_ppl, valid_ppl, test_ppl)
        else:
            batch, label = train_data[i%EPOCH_LEN][0], train_data[i%EPOCH_LEN][1]
            _, ppl = sess.run([train_op, logppl], feed_dict={X: batch, Y: label})
            if i%BATCH_LEN%(BATCH_LEN//10) == 0:
                train_ppl = np.exp(ppl)
                TimeTuple=time.localtime(time.time())
                fmt='%Y-%m-%d %a %H:%M:%S'
                tt=time.strftime(fmt,TimeTuple)
                print '[%s]\t%d\t%.3f\t%.3f' % (tt, i//BATCH_LEN, learning_rate, train_ppl)
    #===Save============================================
            #if i % BATCH_LEN == 0:
            #    final_embeddings = embeddings.eval(session=sess)
            #    word_dict = {}
            #    for id in id2word:
            #        word_dict[id2word[id]] = final_embeddings[id]
            #    outfile = sys.argv[2] + '_' + str(i//BATCH_LEN) + '.txt'
            #    file = open(outfile, 'w')
            #    file.write('%d\n' % VOCAB_SIZE)
            #    file.write('%d\n' % EMBED_SIZE)
            #    for word in word_dict:
            #        file.write('%s\n' % word)
            #        for v in word_dict[word]:
            #            file.write('%lf\n' % v)
            #    file.close()

# vim: set expandtab ts=4 sw=4 sts=4 tw=100
