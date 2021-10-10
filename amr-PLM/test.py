# coding:utf-8

import json
import sys

id_file = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
token_file = json.load(open(sys.argv[2], 'r', encoding='utf-8'))

assert len(id_file) == len(token_file)

for idx in range(len(id_file)):
    id_dict = id_file[idx]
    token_dict = token_file[idx]
    print("".join(token_dict["input_tokens"]))
    # print(id_dict["input_ids"])
    print("".join(token_dict["dec_inp_tokens"]))
    # print(id_dict["dec_inp_ids"])
    print("".join(token_dict["label_tokens"]))
    # print(id_dict["label_ids"])
    print("++++++++++++++++++++++++++++++++++++++")