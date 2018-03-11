# !/usr/bin/python2.7
# -*- coding: utf-8 -*-
 
##############################################################
# 
# Copyright (c) 2018 USTC, Inc. All Rights Reserved
# 
##############################################################
# 
# File:    MY.py
# Author:  roee
# Date:    2018/03/05 22:38:53
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 0 or 1 or 2 ..


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


def my_sru(data, nh, ni, bs, ts):
    limit_u = math.sqrt(6.0 / (ni + nh))
    wei = tf.Variable(tf.random_uniform([ni, nh], -limit_u, limit_u))
    f_x = tf.Variable(tf.random_uniform([ni, nh], -limit_u, limit_u))
    f_b = tf.Variable(tf.zeros([nh]))
    r_x = tf.Variable(tf.random_uniform([ni, nh], -limit_u, limit_u))
    r_b = tf.Variable(tf.zeros([nh]))
    x_t = tf.einsum('ijk,kl>ijl', data, wei)
    f_t = tf.sigmoid(tf.einsum('ijk,kl>ijl', data, f_x) + f_b)
    r_t = tf.sigmoid(tf.einsum('ijk,kl>ijl', data, r_x) + r_b)
    cm = tf.zeros([bs, ts, nh])
    c = tf.unstack(cm, ts, 1)
    sm = tf.zeros([bs, ts, nh])
    s = tf.unstack(sm, ts, 1)
    x = tf.unstack(data, ts, 1)
    _x = tf.unstack(x_t, ts, 1)
    f = tf.unstack(f_t, ts, 1)
    r = tf.unstack(r_t, ts, 1)
    for pos in range(ts):
        if pos == 0:
            c[0] = (1 - f[0]) * _x[0]
            s[0] = r[0] * tf.tanh(c[0]) + (1 - r[0]) * _x[0]
        else:
            c[pos] = f[pos] * c[pos - 1] + (1 - f[pos]) * _x[pos]
            s[pos] = r[pos] * c[pos] + (1 - r[pos]) * _x[pos]
    o = tf.stack(s, 1)
    return o


def my_rnn(data, nh, ni, bs, ts):
    limit_u = math.sqrt(6.0 / (ni + nh))
    limit_v = math.sqrt(6.0 / (2 * nh))
    u = tf.Variable(tf.random_uniform([ni, nh], -limit_u, limit_u))
    w = tf.Variable(tf.random_uniform([nh, nh], -limit_v, limit_v))
    b = tf.Variable(tf.zeros([nh]))
    sm = tf.zeros([bs, ts, nh])
    x = tf.unstack(data, ts, 1)
    s = tf.unstack(sm, ts, 1)
    for pos in range(ts):
        if pos == 0:
            s[0] = tf.tanh(tf.matmul(x[0], u) + b)
        else:
            s[pos] = tf.tanh(tf.matmul(x[pos], u) + tf.matmul(s[pos - 1], w) + b)
    o = tf.stack(s, 1)
    return o

            
def LM(embeddings, X):
    x = tf.reshape(tf.nn.embedding_lookup(embeddings, 
                tf.reshape(X, [BATCH_SIZE * WIN_NUM])), [BATCH_SIZE, WIN_NUM, EMBED_SIZE])
    o = my_sru(x, OUT_SIZE, EMBED_SIZE, BATCH_SIZE, WIN_NUM)
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
    embeddings = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], 
                stddev=1.0/math.sqrt(EMBED_SIZE)))
    output = tf.reshape(LM(embeddings, X), [-1, OUT_SIZE])
    loss_weights = tf.Variable(tf.truncated_normal([VOCAB_SIZE, OUT_SIZE], 
                stddev=1.0/math.sqrt(OUT_SIZE)))
    loss_biases = tf.Variable(tf.truncated_normal([VOCAB_SIZE], 
                stddev=1.0/math.sqrt(VOCAB_SIZE)))
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
