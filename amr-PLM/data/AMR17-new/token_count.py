# coding:utf-8

import sys

with open(sys.argv[1], 'r', encoding='utf-8') as fin:
    data = fin.readlines()
    lengths = [len(line.strip().split()) for line in data]
    max_len, avg_len = max(lengths), sum(lengths)/len(lengths)
    print(f'Max_len: {max_len}, avg_len: {avg_len}')
    for ll in range(1, max_len//100+1):
        mlen = sum([1 for itm in lengths if itm > ll*100])
        print(f'len>{ll*100}: {mlen}')