# coding:utf-8
import sys
import json
import random
random.seed(42)

with open(sys.argv[1]+"/val_linearized_tokens.json", 'r', encoding='utf-8') as fin:
    val_amr_full = json.load(fin)

with open(sys.argv[1]+"/val_tgt_tokens.json", 'r', encoding='utf-8') as fin:
    val_text_full = json.load(fin)

val_gold_amr = []
with open(sys.argv[1]+"/val-gold.amr", 'r', encoding='utf-8') as fin:
    data = fin.readlines()

ith_amr = []
for line in data[2:]:
    if line.startswith("# ::"):
        ith_amr.append(line.rstrip())
    else:
        if line.startswith("# AMR release"):
            continue
        if line.strip() == "":
            if len(ith_amr) > 0:
                val_gold_amr.append("\n".join(ith_amr) + "\n")
            ith_amr = []
        else:
            ith_amr.append(line.rstrip())

assert len(val_amr_full) == len(val_text_full) and len(val_text_full) == len(val_gold_amr)

idxs = list(range(len(val_amr_full)))
random.shuffle(idxs)

train_val_num = int(sys.argv[3])
train_amr = [val_amr_full[idx] for idx in idxs[:train_val_num]]
train_txt = [val_text_full[idx] for idx in idxs[:train_val_num]]

val_amr = [val_amr_full[idx] for idx in idxs[train_val_num:2*train_val_num]]
val_txt = [val_text_full[idx] for idx in idxs[train_val_num:2*train_val_num]]
val_gold = [val_gold_amr[idx] for idx in idxs[train_val_num:2*train_val_num]]

with open(sys.argv[2]+"/train_linearized_tokens.json", 'w', encoding='utf-8') as fout:
    json.dump(train_amr, fout, indent=4)
with open(sys.argv[2]+"/train_tgt_tokens.json", 'w', encoding='utf-8') as fout:
    json.dump(train_txt, fout, indent=4)

with open(sys.argv[2]+"/val_linearized_tokens.json", 'w', encoding='utf-8') as fout:
    json.dump(val_amr, fout, indent=4)
with open(sys.argv[2]+"/val_tgt_tokens.json", 'w', encoding='utf-8') as fout:
    json.dump(val_txt, fout, indent=4)

with open(sys.argv[2]+"/val-gold.amr", 'w', encoding='utf-8') as fout:
    fout.write("\n".join(val_gold) + "\n")