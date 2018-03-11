# !/usr/bin/python2.7
# -*- coding: utf-8 -*-
 
##############################################################
# 
# Copyright (c) 2018 USTC, Inc. All Rights Reserved
# 
##############################################################
# 
# File:    LSTMLM.py
# Author:  roee
# Date:    2018/02/27 10:44:36
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

def my_lstm(data, nh, ni, bs, ts):
    limit_u = math.sqrt(6.0 / (ni + nh))
    limit_v = math.sqrt(6.0 / (2 * nh))
    i_x = tf.Variable(tf.random_uniform([ni, nh], -limit_u, limit_u))
    i_s = tf.Variable(tf.random_uniform([nh, nh], -limit_v, limit_v))
    i_b = tf.Variable(tf.zeros([nh]))

    f_x = tf.Variable(tf.random_uniform([ni, nh], -limit_u, limit_u))
    f_s = tf.Variable(tf.random_uniform([nh, nh], -limit_v, limit_v))
    f_b = tf.Variable(tf.zeros([nh]))

    c_x = tf.Variable(tf.random_uniform([ni, nh], -limit_u, limit_u))
    c_s = tf.Variable(tf.random_uniform([nh, nh], -limit_v, limit_v))
    c_b = tf.Variable(tf.zeros([nh]))

    o_x = tf.Variable(tf.random_uniform([ni, nh], -limit_u, limit_u))
    o_s = tf.Variable(tf.random_uniform([nh, nh], -limit_v, limit_v))
    o_b = tf.Variable(tf.zeros([nh]))

    sm = tf.zeros([bs, ts, nh])
    cm = tf.zeros([bs, ts, nh])
    s = tf.unstack(sm, ts, 1)
    c = tf.unstack(cm, ts, 1)
    x = tf.unstack(data, ts, 1)

    for pos in range(ts):
        if pos == 0:
            c[0] = tf.sigmoid(tf.matmul(x[0], i_x) + i_b) * tf.tanh(tf.matmul(x[0], c_x) + c_b)
            s[0] = tf.sigmoid(tf.matmul(x[0], o_x) + o_b) * tf.tanh(c[0])
        else:
            c[pos] = tf.sigmoid(tf.matmul(s[pos - 1], f_s) + tf.matmul(x[pos], f_x) + f_b) * c[pos - 1] \
                       + tf.sigmoid(tf.matmul(s[pos - 1], i_s) + tf.matmul(x[pos], i_x) + i_b) \
                       * tf.tanh(tf.matmul(s[pos - 1], c_s) + tf.matmul(x[pos], c_x) + c_b)
            s[pos] = tf.sigmoid(tf.matmul(s[pos - 1], o_s) + tf.matmul(x[pos], o_x) + o_b) * tf.tanh(c[pos])
    o = tf.stack(s, 1)
    return o

            
def LM(embeddings, X):
    x = tf.reshape(tf.nn.embedding_lookup(embeddings, 
                tf.reshape(X, [BATCH_SIZE * WIN_NUM])), [BATCH_SIZE, WIN_NUM, EMBED_SIZE])
    cell = tf.contrib.rnn.BasicLSTMCell(OUT_SIZE)
    h = cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    o, state = tf.nn.dynamic_rnn(cell=cell, inputs=x, initial_state=h)
    #o = my_lstm(x, OUT_SIZE, EMBED_SIZE, BATCH_SIZE, WIN_NUM)
    return o
   

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print 'python '+ sys.argv[0] +' corpus/ptb/ vector/ptb/rnn'
        sys.exit()
    #===Paras============================================
    para = {}
    para['batch_size'] = 100
    para['win_num'] = 5
    para['embed_size'] = 50
    para['hidden_size'] = 50
    para['max_max_epoch'] = 50
    para['vocab_pro'] = 1.0
    BATCH_SIZE = para['batch_size']   
    WIN_NUM = para['win_num']
    EMBED_SIZE = para['embed_size']
    OUT_SIZE = para['hidden_size']
    EPOCH_SIZE = para['max_max_epoch']
    VOCAB_PRO = para['vocab_pro']
    for p in para:
        print p + ':\t' + str(para[p])
    #===Read_corpus============================================
    train_file = open(sys.argv[1]+'train.txt', 'r')
    valid_file = open(sys.argv[1]+'valid.txt', 'r')
    test_file = open(sys.argv[1]+'test.txt', 'r')
    train = file2list(train_file)
    data, count, word2id, id2word = build_dataset(train, VOCAB_PRO) 
    VOCAB_SIZE = len(count)
    print 'All vocabulary size: ', VOCAB_SIZE
    DATA_LEN = len(data)
    BATCH_LEN = DATA_LEN // BATCH_SIZE
    EPOCH_LEN = (BATCH_LEN - 2) // (WIN_NUM)
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
        dat[i] = data[batch_len * i:batch_len * (i+1)]
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
        dat[i] = data[batch_len * i:batch_len * (i+1)]
    valid_data = []
    for i in range(epoch_len):
        x = dat[:, i*num_steps:(i+1)*num_steps]
        y = dat[:, i*num_steps+1:(i+1)*num_steps+1]
        valid_data.append([x, y])
    TRAIN_STEPS = len(data) * EPOCH_SIZE / BATCH_SIZE
    #===Train============================================
    X = tf.placeholder(tf.int32, shape=[BATCH_SIZE, WIN_NUM])
    Y = tf.placeholder(tf.int32, shape=[BATCH_SIZE, WIN_NUM])
    embeddings = tf.get_variable(name='embeddings', shape=[VOCAB_SIZE, EMBED_SIZE],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32))
    output = tf.reshape(LM(embeddings, X), [-1, OUT_SIZE])
    loss_weights = tf.get_variable(name='lose_weights', shape=[VOCAB_SIZE, OUT_SIZE],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32))
    loss_biases = tf.get_variable(name='lose_biases', shape=[VOCAB_SIZE], 
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32))
    logits = tf.matmul(output, tf.transpose(loss_weights)) + loss_biases
    cost = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], 
            [tf.reshape(Y, [-1])], [tf.ones([BATCH_SIZE*(WIN_NUM)], dtype=tf.float32)])
    logppl = tf.reduce_sum(cost) / BATCH_SIZE 
    optimizer = tf.train.AdamOptimizer(0.001).minimize(logppl)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    print 'Epoch size: ', EPOCH_SIZE
    print 'Iterations: ', TRAIN_STEPS
    for i in range(TRAIN_STEPS+1):
        batch, label = train_data[i%EPOCH_LEN][0], train_data[i%EPOCH_LEN][1]
        _, p = sess.run([optimizer, logppl], feed_dict={X: batch, Y: label})
        if (i%BATCH_LEN)%(BATCH_LEN//20) == 0:
            ppl = np.exp(p / len(train_data[0][0][0]))
            costs = 0
            for batch, label in valid_data:
                cost = sess.run(logppl, feed_dict={X: batch, Y: label})
                costs += cost
            valid_ppl = np.exp(costs / len(valid_data) / len(valid_data[0][0][0]))
            costs = 0
            for batch, label in test_data:
                cost = sess.run(logppl, feed_dict={X: batch, Y: label})
                costs += cost
            test_ppl = np.exp(costs / len(test_data) / len(test_data[0][0][0]))
            TimeTuple=time.localtime(time.time())
            fmt='%Y-%m-%d %a %H:%M:%S'
            tt=time.strftime(fmt,TimeTuple)
            print '[%s]\t%d\t%d\t%.3f\t%.3f\t%.3f' % (tt, i//BATCH_LEN, 
                (i%BATCH_LEN)/(BATCH_LEN//20)*5, ppl, valid_ppl, test_ppl)
    #===Save============================================
            if i % BATCH_LEN == 0:
                final_embeddings = embeddings.eval(session=sess)
                word_dict = {}
                for id in id2word:
                    word_dict[id2word[id]] = final_embeddings[id]
                outfile = sys.argv[2] + '_' + str(i//BATCH_LEN) + '.txt'
                file = open(outfile, 'w')
                file.write('%d\n' % VOCAB_SIZE)
                file.write('%d\n' % EMBED_SIZE)
                for word in word_dict:
                    file.write('%s\n' % word)
                    for v in word_dict[word]:
                        file.write('%lf\n' % v)
                file.close()

# vim: set expandtab ts=4 sw=4 sts=4 tw=100
