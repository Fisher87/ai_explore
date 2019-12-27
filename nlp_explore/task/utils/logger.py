#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：logger.py
#   创 建 者：Yulianghua
#   创建日期：2019年12月01日
#   描    述：
#
#================================================================

import os
import logging

def glogger(name, input_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filehandler = logging.FileHandler(input_file, mode="w")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(message)s')
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return logger
