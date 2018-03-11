# !/usr/bin/python2.7
# -*- coding: utf-8 -*-
 
##############################################################
# 
# Copyright (c) 2018 USTC, Inc. All Rights Reserved
# 
##############################################################
# 
# File:    Skip-gram.py
# Author:  roee
# Date:    2018/02/01 15:34:28
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
import re
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 0 or 1 or 2 ..


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip()


def build_dataset(words, vocab, vocabulary_size):
    count = [['<UNK>', 0]]
    count.extend(vocab.most_common(vocabulary_size - 1))
    word2id = {}
    for word, _ in count:
        word2id[word] = len(word2id)
    data = []
    unk_count = 0
    for word in words:
        if word in word2id:
            index = word2id[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    id2word = dict(zip(word2id.values(), word2id.keys()))
    return data, count, word2id, id2word


def skip_next_batch(data, batch_size, win_num):
    global data_index
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    label = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    n = win_num - 1
    for i in range(batch_size/n):
        if (data_index + win_num) > len(data):
            data_index = 0
        temp = data[data_index:(data_index + win_num)]
        aim = temp[n/2]
        temp.remove(aim)
        random.shuffle(temp)
        for j in range(len(temp)):
            batch[n*i+j] = aim
            label[n*i+j] = [temp[j]]
        data_index = (data_index + 1) % len(data)
    return batch, label

    
def Skip_gram(embeddings, X):
    x = tf.nn.embedding_lookup(embeddings, tf.reshape(X, [BATCH_SIZE]))
    return x
   

if __name__ == "__main__":
    #===Paras============================================
    VOCAB_SIZE_PRO = 0.9
    BATCH_SIZE = 100
    WIN_NUM = int(sys.argv[2]) # odd
    EMBED_SIZE = 50
    OUT_SIZE = EMBED_SIZE
    #SAMPLE_SIZE = 10
    #SIMILAR_SIZE = 5
    EPOCH_SIZE = int(sys.argv[3])
    SAMPLE_NUM = 100
    #===Read_corpus============================================
    train_file = open(sys.argv[1]+'train.txt', 'r')
    valid_file = open(sys.argv[1]+'valid.txt', 'r')
    test_file = open(sys.argv[1]+'test.txt', 'r')
    train = []
    for line in train_file:
        sentence = clean_str(line).split()
        train.extend(sentence)
    vocab = collections.Counter(train)
    VOCAB_SIZE = int(len(vocab) * VOCAB_SIZE_PRO)
    print 'All vocabulary size: ', len(vocab)
    data, count, word2id, id2word = build_dataset(train, vocab, VOCAB_SIZE) 
    test = []
    for line in test_file:
        sentence = clean_str(line).split()
        sen = []
        for word in sentence:
            if word in word2id:
                sen.append(word2id[word])
            else:
                sen.append(word2id['<UNK>'])
        test.extend(sen)
    valid = []
    for line in valid_file:
        sentence = clean_str(line).split()
        sen = []
        for word in sentence:
            if word in word2id:
                sen.append(word2id[word])
            else:
                sen.append(word2id['<UNK>'])
        valid.extend(sen)
    TRAIN_STEPS = len(data) * EPOCH_SIZE / BATCH_SIZE
    #sample_list = random.sample(range(VOCAB_SIZE), SAMPLE_SIZE)
    #===Train============================================
    X = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    Y = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
    embeddings = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], stddev=1.0/math.sqrt(EMBED_SIZE)))
    output = Skip_gram(embeddings, X)
    loss_weights = tf.Variable(tf.truncated_normal([VOCAB_SIZE, OUT_SIZE], stddev=1.0/math.sqrt(OUT_SIZE)))
    loss_biases = tf.Variable(tf.truncated_normal([VOCAB_SIZE], stddev=1.0/math.sqrt(VOCAB_SIZE)))
    logits = tf.matmul(output, tf.transpose(loss_weights)) + loss_biases
    labels_one_hot = tf.one_hot(Y, VOCAB_SIZE)
    logppl = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits))
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=loss_weights, biases=loss_biases, inputs=output, labels=Y, num_sampled=SAMPLE_NUM, num_classes=VOCAB_SIZE))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    print 'Epoch size: ', EPOCH_SIZE
    print 'Iterations: ', TRAIN_STEPS
    data_index = 0
    for i in range(TRAIN_STEPS+1):
        batch, label = skip_next_batch(data, BATCH_SIZE, WIN_NUM)
        _, lo, p = sess.run([optimizer, loss, logppl], feed_dict={X: batch, Y: label})
        if i % 1000 == 0:
            index_temp = data_index
            data_index = 0
            costs = 0
            num = len(test) / BATCH_SIZE
            for _ in range(num):
                batch, label = skip_next_batch(test, BATCH_SIZE, WIN_NUM)
                cost = sess.run(logppl, feed_dict={X: batch, Y: label})
                costs += cost
            test_ppl = np.exp(costs / num)
            data_index = index_temp

            index_temp = data_index
            data_index = 0
            costs = 0
            num = len(valid) / BATCH_SIZE
            for _ in range(num):
                batch, label = skip_next_batch(valid, BATCH_SIZE, WIN_NUM)
                cost = sess.run(logppl, feed_dict={X: batch, Y: label})
                costs += cost
            valid_ppl = np.exp(costs / num)
            data_index = index_temp
            print 'iter: %d\tloss: %.3lf\tvalid ppl: %.3lf\ttest ppl: %.3lf' % (i, p, valid_ppl, test_ppl)
    final_embeddings = embeddings.eval(session=sess)
    #===Save============================================
    word_dict = {}
    for id in id2word:
        word_dict[id2word[id]] = final_embeddings[id]
    file = open(sys.argv[4], 'w')
    file.write('%d\n' % VOCAB_SIZE)
    file.write('%d\n' % EMBED_SIZE)
    for word in word_dict:
        file.write('%s\n' % word)
        for v in word_dict[word]:
            file.write('%lf\n' % v)
    file.close()

# vim: set expandtab ts=4 sw=4 sts=4 tw=100
