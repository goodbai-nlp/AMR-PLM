from transformers import T5Tokenizer
def build_mapping(sent1, sent2):
    i, j = 0, 0
    new_sent = []
    flag = False
    while i < len(sent1):
        word = sent1[i]
        start = j
        curstr = sent2[j]
        while start < len(sent2) and curstr != word:
            new_sent.append(sent2[start] + "@@")
            start += 1
            if start >= len(sent2) and curstr != word:
                print("word:{} could not be replaced by {} !!!!".format(word, sent2[j:]))
                flag = True
            else:
                curstr = curstr + sent2[start]
        if flag:
            print("Inconsistent sents orr:{}, toked:{}".format(sent1, sent2))
            break
        else:
            new_sent.append(sent2[start])
            j = start + 1
            i += 1
    return " ".join(new_sent)


line="multi-sentence :snt1 many :ARG0-of sense :ARG1 urgency :time watch :ARG0 :ARG1 thing :manner-of develop :ARG0 thing :manner quiet :ARG1 :snt2 dragon :domain you :ARG0-of coil :snt3 tiger :domain you :ARG0-of crouch :snt4 admire :ARG0 i :ARG1 patriot :ARG0-of mind :mod noble ."
spm="▁multi - sent ence ▁support ▁imperative ▁you ▁person ▁start ▁thread ▁ re sol ute ▁reply ▁ i ▁compose ▁ i ▁poem"
# special_prefix = "▁"
# line_reconstructed =''.join([itm.replace(special_prefix, " ", 1) for itm in spm.split()])
# line_bpe = spm.replace(' ', '@@').replace(special_prefix, ' ')
# print("line_ori", line)
# print("line_rec", line_reconstructed)
# print("line_bpe", line_bpe)

tokenizer = T5Tokenizer.from_pretrained("t5-large")
tmp1 = tokenizer(line, truncation=False)
print(len(tmp1['input_ids']), tmp1['input_ids'])
tmp2 = tokenizer(line.split(), is_split_into_words=True, truncation=False)
print(len(tmp2['input_ids']), tmp2['input_ids'])

# special_prefix = "▁"
# # line_tokenized = [itm.replace(special_prefix, "", 1) for itm in spm.split()]
# line_tokenized = spm.replace(special_prefix, '').split()
# print("Tokenized:", line_tokenized)
# line_new = build_mapping(line.split(), line_tokenized)
# print(line_new)