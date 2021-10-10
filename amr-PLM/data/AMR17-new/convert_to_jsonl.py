# coding:utf-8

import sys
import json

with open(sys.argv[1], "r", encoding="utf-8") as fin:
    data1 = fin.readlines()

with open(sys.argv[2], "r", encoding="utf-8") as fin:
    data2 = fin.readlines()

assert len(data1) == len(data2)

json_lines = [
    json.dumps({
        "src": data1[idx].strip(),
        "tgt": data2[idx].strip(),
    })
    for idx in range(len(data1)) if "http" not in data1[idx].strip()
]

with open(sys.argv[3], "w", encoding="utf-8") as fout:
    fout.write("\n".join(json_lines))
