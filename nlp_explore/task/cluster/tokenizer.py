#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：segment.py
#   创 建 者：YuLianghua
#   创建日期：2020年05月08日
#   描    述：
#
#================================================================

import pdb
import os
import json
import codecs
from collections import defaultdict
from collections import Counter

import pkuseg
import spacy

SPECIAL_SIMBLE = {'=', '/', '[', ']', '{', '}', '(', ')', ':'}

def read_dict(f):
    with codecs.open(f, 'r', 'utf8') as rf:
        return set([w.strip() for w in rf.readlines()])

class TokenHandler(object):
    def __init__(self, config, norm=True, postag=True, **kwargs):
        """
        @param: user_dict
        """
        self.config = config
        self.norm   = norm
        self.postag = postag
        self.en_handler = spacy.load("en_core_web_sm")
        self.seghandler = pkuseg.pkuseg(user_dict=config['tokenizer']['user_dict'],
                                        postag=postag, **kwargs)
        self.init()

    def init(self):
        self.user_dict = set([_.strip() for _ in open(self.config['tokenizer']['user_dict'], 'r').\
                              readlines() if _])
        char_map_list  = [l.strip().split('\t') for l in open(self.config['tokenizer']['char_map'], 'r').\
                          readlines() if l]
        self.char_map  = dict(char_map_list)

    def normp(self, txt):
        for c,v in self.char_map.items():
            if c in self.char_map:
                txt = txt.replace(c, v)
        return txt

    def seg_postprocess(self, query, seg_res):
        def char_type(c):
            is_chinese = lambda x: '\u4e00' <= x <= '\u9fa5'
            return 'C' if is_chinese(c) else 'E'

        def is_alphad(w):
            if w in ('_', '-', '.'):
                return True
            elif w.isdigit():
                return True
            else:
                try:
                    return w.encode("ascii").isalpha()
                except:
                    return False

        def token_type(token):
            '''check token is alphas or digits'''
            if all(is_alphad(t) for t in token) :
                return 'alphad'
            else:
                return 'other'

        new_seg_res = []
        for s in seg_res:
            if len(s)==1 or s in self.user_dict:
                new_seg_res.append(s)
            else:
                ss = 0
                while ss<len(s) and s[ss] in SPECIAL_SIMBLE:
                    new_seg_res.append(s[ss])
                    ss += 1
                if ss>=len(s):
                    continue
                ns = s[ss:]
                cs = ns[0]
                c_type = char_type(cs)
                i=1
                while i < len(ns):
                    _c = ns[i]
                    _c_type = char_type(_c)
                    # 出现特殊符号则进行分割
                    if _c in SPECIAL_SIMBLE:
                        new_seg_res.append(cs)
                        new_seg_res.append(_c)
                        i += 1
                        cs = ''
                        continue
                    if _c_type == c_type:
                        cs+=_c
                    else:
                        new_seg_res.append(cs)
                        c_type = _c_type 
                        cs = _c
                    i+=1
                new_seg_res.append(cs)

        start, end = 0, 0
        seg_infos = []
        for s in new_seg_res:
            _ = defaultdict(dict)
            start = end
            end = start + len(s)
            while(query[start:end]!=s):
                start += 1
                end = start + len(s)
            _[s]['pos'] = (start, end)
            _[s]['type']= token_type(s)
            seg_infos.append(_)

        assert len(new_seg_res) == len(seg_infos)
        # print(json.dumps(seg_infos, ensure_ascii=False))
        
        segments = []
        seg_pos  = []
        i = 0
        while(i<len(seg_infos)):
            new_seg_t = new_seg_res[i]
            c_start= seg_infos[i][new_seg_t]['pos'][0]
            c_end  = seg_infos[i][new_seg_t]['pos'][1]
            c_type = seg_infos[i][new_seg_t]['type']
            t = 1
            if c_type == "other" or (i==len(seg_infos)-1):
                segments.append(new_seg_t)
                seg_pos.append((c_start, c_end))
                i = i+t
                continue
            seg_length = len(seg_infos)
            while(t<seg_length-i):
                n_seg_t = new_seg_res[i+t]
                n_start = seg_infos[i+t][n_seg_t]['pos'][0]
                n_end   = seg_infos[i+t][n_seg_t]['pos'][1]
                n_type  = seg_infos[i+t][n_seg_t]['type']
                if n_type == "other":
                    segments.append(new_seg_t)
                    seg_pos.append((c_start, c_end))
                    i = i+t
                    break
                else:
                    if c_end == n_start:
                        new_seg_t += n_seg_t
                        c_end = n_end
                        t += 1
                        if (i+t == (seg_length)):
                            segments.append(new_seg_t)
                            seg_pos.append((c_start, c_end))
                            i = i+t
                            break
                    else:
                        segments.append(new_seg_t)
                        seg_pos.append((c_start, c_end))
                        i = i+t
                        break
        return segments, seg_pos

    def seg_map(self, seg_pos, ori_query):
        '''输出还原(大小写) 
        '''
        new_seg = []
        for start,end in seg_pos:
            new_seg.append(ori_query[start:end])
            start = end

        return new_seg

    def seg(self, txt, lang='zh'):
        '''
        @param txt: string, query to be segmented;
        @param lang: lang type('zh'|'en'), (default:`zh`)
        '''
        if lang=='zh':
            char_postag = dict()
            if self.norm: 
                ntxt = self.normp(txt)
            txt_lower = ntxt.lower()
            if not self.postag:
                _seg = self.seghandler.cut(txt_lower)
            else:
                seg_postag = self.seghandler.cut(txt_lower)
                _seg = [sp[0] for sp in seg_postag]
                segwords_str = ''.join(_seg)
                postag_list = []
                for _s,_p in seg_postag:
                    postag_list += [_p]*len(_s) 
                j = 0
                for i in range(len(txt_lower)):
                    if txt_lower[i] == segwords_str[j]:
                        char_postag[i] = postag_list[j]
                        j += 1

            seg, seg_pos = self.seg_postprocess(txt_lower, _seg)
            if txt_lower!=txt:
                seg = self.seg_map(seg_pos, txt)
            if self.postag:
                _seg_postag = []
                for i,s_pos in enumerate(seg_pos):
                    c_postag = [char_postag[i] for i in range(s_pos[0], s_pos[1])]
                    c_postag_most = Counter(c_postag).most_common(1)
                    _seg_postag.append((seg[i], c_postag_most[0][0]))
                seg = _seg_postag
        else:
            # lang 'en'
            start = time.time()
            txt_tokens = self.en_handler(txt) 
            if not self.postag:
                seg = [t.text for t in txt_tokens]
            else:
                seg = [(t.text, t.pos_) for t in txt_tokens]

        return seg

    def seg_file(self, input_f, output_f, **kwargs):
        """
        @param: nthread
        """
        return pkuseg.test(input_f, output_f, **kwargs)

    def train(self, train_f, test_f, model_save_f, **kwargs):
        """
        @param: train_iter
        @param: init_model
        """
        return pkuseg.train(train_f, test_f, model_save_f, **kwargs)

