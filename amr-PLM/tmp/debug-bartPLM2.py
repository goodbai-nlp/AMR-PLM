# coding:utf-8
import torch

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
hf_config = BartConfig.from_pretrained('facebook/bart-large')
hf_config.force_bos_token_to_be_generated = True
tok = BartTokenizer.from_pretrained("facebook/bart-large")
hf_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large', config=hf_config)

# fsq bart base
# from fairseq.models.bart import BARTModel
# bart = BARTModel.from_pretrained('/nfs/Desktop/data/pretrained-model/bart.large', checkpoint_file='model.pt')

# mask_token_id = bart.task.source_dictionary.indices["<mask>"]
# mask_token_weight_fairseq = bart.model.encoder.embed_tokens.weight[mask_token_id].detach()

# mask_token_id_hf = tok.mask_token_id
# mask_token_weight_hf = hf_model.model.encoder.embed_tokens.weight[mask_token_id_hf].detach()

# print((mask_token_weight_hf - mask_token_weight_fairseq).abs().max())

# input_string = " My dog is <mask>"
# labels_string = " My dog is cute"
input_string = "Rudolph Agnew <mask> 55 years old and former chairman of <mask> ated <mask> <mask> PLC , was named a nonexecutive director of this British industrial conglomerate ."
labels_string = "Rudolph Agnew , 55 years old and former chairman of Consolidated Gold Fields PLC , was named a nonexecutive director of this British industrial conglomerate ."

input_ids = tok(input_string, return_tensors="pt").input_ids
labels = tok(labels_string, return_tensors="pt").input_ids
# print("labels", labels)
# print(tok.batch_decode(labels.tolist()))
decoder_input_ids = hf_model.prepare_decoder_input_ids_from_labels(labels)
print("input_ids", input_ids)
print(tok.batch_decode(input_ids.tolist()))
print("dec_inp_ids", decoder_input_ids)
print(tok.batch_decode(decoder_input_ids.tolist()))
print("labels", labels)
print(tok.batch_decode(labels.tolist()))
loss = hf_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)[0]
print("Loss1", loss)

# from transformers import BartTokenizer, BartForConditionalGeneration
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

# TXT = "My friends are <mask> but they eat too many carbs."
# input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
# print(input_ids)
# logits = model(input_ids).logits

# masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
# probs = logits[0, masked_index].softmax(dim=0)
# values, predictions = probs.topk(5)
# print(tokenizer.decode(predictions).split())
