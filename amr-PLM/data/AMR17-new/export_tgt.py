# coding:utf-8

import json
import sys
with open(sys.argv[1], 'r', encoding='utf-8') as fin:
    data = json.load(fin)
    text_data = '\n'.join(data)

    with open(sys.argv[2], 'w', encoding='utf-8') as fout:
        fout.write(text_data)
