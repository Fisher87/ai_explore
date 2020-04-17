#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：tools.py
#   创 建 者：YuLianghua
#   创建日期：2020年04月16日
#   描    述：
#
#================================================================

def edit_distance(s1, s2):
    """caculate edit distance between two strings;
    """
    edit = [[i+j for j in range(len(s2)+1)] for i in range(len(s1)+1)]

    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            if s1[i-1]==s2[j-1]:
                d = 0
            else:
                d = 1

            edit[i][j] = min(edit[i-1][j]+1, edit[i][j-1]+1, edit[i-1][j-1]+d)
    return edit[len(s1)][len(s2)]


if __name__ == "__main__":
    ed = edit_distance('hello', 'hexxol')
    print(ed)

