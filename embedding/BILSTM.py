# !/usr/bin/python2.7
# -*- coding: utf-8 -*-
 
##############################################################
# 
# Copyright (c) 2018 USTC, Inc. All Rights Reserved
# 
##############################################################
# 
# File:    TESTLM.py
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 0 or 1 or 2 ..


def file2list(input_file):
    output_list = []
    for line in input_file:
        sentence = line.replace('\n', ' <eos> ').split()
        output_list.extend(sentence)
    return output_list

def build_dataset(words, vocab, vocabulary_size):
    count = [['<eos>', 0]]
    count.extend(vocab.most_common(vocabulary_size - 1))
    word2id = {}
    for word, _ in count:
        word2id[word] = len(word2id)
    data = []
    for word in words:
        index = word2id[word]
        data.append(index)
    id2word = dict(zip(word2id.values(), word2id.keys()))
    return data, count, word2id, id2word


def my_gru(data, nh, ni, bs, ts):
    limit_u = math.sqrt(6.0 / (ni + nh))
    limit_v = math.sqrt(6.0 / (2 * nh))
    z_x = tf.Variable(tf.random_uniform([ni, nh], -limit_u, limit_u))
    z_s = tf.Variable(tf.random_uniform([nh, nh], -limit_v, limit_v))

    r_x = tf.Variable(tf.random_uniform([ni, nh], -limit_u, limit_u))
    r_s = tf.Variable(tf.random_uniform([nh, nh], -limit_v, limit_v))

    t_x = tf.Variable(tf.random_uniform([ni, nh], -limit_u, limit_u))
    t_s = tf.Variable(tf.random_uniform([nh, nh], -limit_v, limit_v))

    sm = tf.zeros([bs, ts, nh])
    s = tf.unstack(sm, ts, 1)
    x = tf.unstack(data, ts, 1)

    for pos in range(ts):
        if pos == 0:
            z = tf.sigmoid(tf.matmul(x[0], z_x))
            t = tf.tanh(tf.matmul(x[0], t_x))
            s[pos] = z * t
        else:
            z = tf.sigmoid(tf.matmul(s[pos - 1], z_s) + tf.matmul(x[pos], z_x))
            r = tf.sigmoid(tf.matmul(s[pos - 1], r_s) + tf.matmul(x[pos], r_x))
            t = tf.tanh(tf.matmul(r * s[pos - 1], t_s) + tf.matmul(x[pos], t_x))
            s[pos] = (1 - z) * s[pos - 1] + z * t
    o = tf.stack(s, 1)
    return o, o


def LM(embeddings, X):
    x = tf.reshape(tf.nn.embedding_lookup(embeddings, 
                tf.reshape(X, [BATCH_SIZE * WIN_NUM])), [BATCH_SIZE, WIN_NUM, EMBED_SIZE])
    x = tf.unstack(x, WIN_NUM, 1)
    fw_cell = tf.contrib.rnn.BasicLSTMCell(OUT_SIZE)
    bw_cell = tf.contrib.rnn.BasicLSTMCell(OUT_SIZE)
    O, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
    output = tf.stack(O, 1)
    return output
   

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print 'python '+ sys.argv[0] +' corpus/ptb/ vector/ptb/rnn'
        sys.exit()
    #===Paras============================================
    BATCH_SIZE = 100
    WIN_NUM = 5
    EMBED_SIZE = 50
    OUT_SIZE = 50
    EPOCH_SIZE = 50
    #===Read_corpus============================================
    train_file = open(sys.argv[1]+'train.txt', 'r')
    valid_file = open(sys.argv[1]+'valid.txt', 'r')
    test_file = open(sys.argv[1]+'test.txt', 'r')
    train = file2list(train_file)
    vocab = collections.Counter(train)
    VOCAB_SIZE = len(vocab) + 1
    print 'All vocabulary size: ', len(vocab)
    data, count, word2id, id2word = build_dataset(train, vocab, VOCAB_SIZE) 
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
    output = tf.reshape(LM(embeddings, X), [-1, 2*OUT_SIZE])
    loss_weights = tf.Variable(tf.truncated_normal([VOCAB_SIZE, 2*OUT_SIZE], 
                stddev=1.0/math.sqrt(2*OUT_SIZE)))
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
