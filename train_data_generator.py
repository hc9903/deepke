#@title
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 23:20:51 2020

@author: huochao
"""


isa_count=0
unknown_count=0
tags={'BaiduCARD', 'BaiduTAG', '职业', '职称','描述','标签'}
d = dict()
f=open('data/baike_triples.txt',encoding="utf8")
b =open('data/big.txt','w',encoding="utf8")
count = 0
for i in f:
    count += 1
    if count % 1000000 == 0:
        for e in d:
            if 'BaiduCARD' not in d[e]:
                continue
            sentence = d[e]['BaiduCARD'].pop()
            sentence = sentence.replace("<a>",'').replace("</a>",'')
            for t in d[e]:
                if t != 'BaiduCARD':
                    for tail in d[e][t]:
                        if e == tail:
                            continue
                        if e in sentence and tail in sentence:
                            if tail in 'head':
                              continue
                            head_start = sentence.find(e)
                            head_end = head_start+len(e)-1
                            
                            tail_start = sentence.find(tail)
                            tail_end = tail_start+len(tail)-1
                            
                            head_offset = str(head_start)
                            tail_offset=str(tail_start)
                            
                            if head_start>=tail_start and head_start <=tail_end:
                                continue
                            if head_end>=tail_start and head_end <=tail_end:
                                continue
                            if tail_start>=head_start and tail_start<=head_end:
                                continue
                            if tail_end>=head_start and tail_end<=head_end:
                                continue


                            r = 'unknown'
                            if t in tags:
                                r='is-a'
                                if isa_count<=unknown_count:
                                    b.write(sentence+'$&$'+r+'$&$'+e+'$&$'+head_offset+'$&$'+tail+'$&$'+tail_offset)
                                    b.write('\n')
                                    isa_count += 1
                            
                            
                            if r=='unknown' and unknown_count<=isa_count:
                                b.write(sentence+'$&$'+r+'$&$'+e+'$&$'+head_offset+'$&$'+tail+'$&$'+tail_offset)
                                b.write('\n')
                                unknown_count += 1
        d = dict()
    l=i.strip().split('\t')
    if l[0] not in d:
        d[l[0]] = dict()
    if l[1] not in d[l[0]]:
        d[l[0]][l[1]]=set()
    d[l[0]][l[1]].add(l[2])

b.close()
f.close()






b =open('data/big.txt',encoding="utf8")
train=open("data/origin/train.csv",'w',encoding="utf8")
valid = open("data/origin/valid.csv",'w',encoding="utf8")
test =open("data/origin/test.csv",'w',encoding="utf8")
count=0
train.write('sentence$&$relation$&$head$&$head_offset$&$tail$&$tail_offset\n')
valid.write('sentence$&$relation$&$head$&$head_offset$&$tail$&$tail_offset\n')
test.write('sentence$&$relation$&$head$&$head_offset$&$tail$&$tail_offset\n')
for i in b:
    if count%10 < 3:
        train.write(i)
    if count%10 == 6:
        valid.write(i)
    if count%10 == 9:
        test.write(i)
    count += 1
train.close()
valid.close()
test.close()
b.close()