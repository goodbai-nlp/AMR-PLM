from transformers import BartTokenizer, BartTokenizerFast

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
tokenizer.add_tokens(['Salah', 'ĠSalah'])


# new_tokens_vocab = {}
# new_tokens_vocab['additional_special_tokens'] = ['ĠSalah']
# tokenizer.add_special_tokens(new_tokens_vocab)
# INIT = 'Ġ'
# tokens = [INIT+'Salah']
# old_enc_size = len(tokenizer.encoder)
# print(old_enc_size)
# for i, t in enumerate(tokens, start=old_enc_size):
#     tokenizer.encoder[t] = i
# tokenizer.encoder = {k: i for i, (k,v) in enumerate(sorted(tokenizer.encoder.items(), key=lambda x: x[1]))}
# tokenizer.decoder = {v: k for k, v in sorted(tokenizer.encoder.items(), key=lambda x: x[1])}
print(len(tokenizer))
print(tokenizer.tokenize('I love Salah and salad'))
print(tokenizer('I love Salah and salad'))
print(tokenizer.tokenize('I love Mike and salad'))
print(tokenizer('I love Mike and salad'))
print(tokenizer.decode(tokenizer('I love Salah and salad')['input_ids']))
'''
tokenizer_fast = BartTokenizerFast.from_pretrained('facebook/bart-large')
tokenizer_fast.add_tokens(['Salah'])
print('############ sep #############')
print(tokenizer_fast.tokenize('I love Salah and salad'))
print(tokenizer_fast('I love Salah and salad'))
print(tokenizer_fast.decode(tokenizer_fast('I love Salah and salad')['input_ids']))
'''
