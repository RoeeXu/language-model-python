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



for path in `ls ../../vector/wsj/`
do
echo './senna_tag ../../vector/wsj/'$path' &> eval/'$path'.eval &'
done













# vim: set expandtab ts=4 sw=4 sts=4 tw=100
