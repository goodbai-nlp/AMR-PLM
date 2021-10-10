# coding:utf-8

import json
import sys
init='Ġ'
with open(sys.argv[1], 'r', encoding='utf-8') as fin:
    data = json.load(fin)
    # text_data = '\n'.join(' '.join(line[1:-1]).replace(init, '') for line in data)
    #text_data = '\n'.join(' '.join(line).replace(init, '') for line in data)
    text_data = '\n'.join(' '.join(line) for line in data)

    with open(sys.argv[2], 'w', encoding='utf-8') as fout:
        fout.write(text_data)
