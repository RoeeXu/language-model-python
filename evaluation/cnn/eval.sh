# !/bin/bash
 
##############################################################
# 
# Copyright (c) 2018 USTC, Inc. All Rights Reserved
# 
##############################################################
# 
# File:    eval.sh
# Author:  roee
# Date:    2018/02/26 09:09:28
# Brief:
# 
# 
##############################################################



for i in {0..4}
do
for path in `ls ../../vector/sst/`
do
echo './cnn_senna ../../vector/sst/'$path' tree_train.txt tree_test.txt 5 '$i' tree_dev.txt 5 90 &> eval/'$path'_'$i'.eval &'
done
done













# vim: set expandtab ts=4 sw=4 sts=4 tw=100
