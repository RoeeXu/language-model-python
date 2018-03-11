# !/bin/bash
 
##############################################################
# 
# Copyright (c) 2018 USTC, Inc. All Rights Reserved
# 
##############################################################
# 
# File:    train.sh
# Author:  roee
# Date:    2018/02/25 17:25:08
# Brief:
# 
# 
##############################################################

cop=$1
epoch=$2
echo 'python embedding/NNLM.py corpus/'${cop}'/ 5 '${epoch}' vector/'${cop}'/NNLM_5_'${epoch}'.txt &> log/'${cop}'/log_NNLM_5_'${epoch}'.txt &'
echo 'python embedding/Skip-gram.py corpus/'${cop}'/ 5 '${epoch}' vector/'${cop}'/Skip_5_'${epoch}'.txt &> log/'${cop}'/log_Skip_5_'${epoch}'.txt &'
echo 'python embedding/CBOW.py corpus/'${cop}'/ 5 '${epoch}' vector/'${cop}'/CBOW_5_'${epoch}'.txt &> log/'${cop}'/log_CBOW_5_'${epoch}'.txt &'
echo 'python embedding/RNNLM.py corpus/'${cop}'/ 35 '${epoch}' vector/'${cop}'/RNNLM_35_'${epoch}'.txt &> log/'${cop}'/log_RNNLM_35_'${epoch}'.txt &'
echo 'python embedding/ABLLM.py corpus/'${cop}'/ 35 '${epoch}' vector/'${cop}'/ABLLM_35_'${epoch}'.txt &> log/'${cop}'/log_ABLLM_35_'${epoch}'.txt &'




















# vim: set expandtab ts=4 sw=4 sts=4 tw=100
