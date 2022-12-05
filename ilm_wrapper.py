#wget https://github.com/chrisdonahue/ilm/archive/master.zip
#unzip master.zip
#pip3 install -e ilm-master/
#pip3 install --upgrade transformers==2.7.0

#python3 ilm-master/acl20_repro.py model sto ilm | bash

MASK_CLS = 'ilm.mask.hierarchical.MaskHierarchical'
MODEL_DIR = '/tmp/ilm/models/sto_ilm'

from ilm.infer import infill_with_ilm

import os
import pickle

import ilm.tokenize_util

tokenizer = ilm.tokenize_util.Tokenizer.GPT2
with open(os.path.join(MODEL_DIR, 'additional_ids_to_tokens.pkl'), 'rb') as f:
    additional_ids_to_tokens = pickle.load(f)
additional_tokens_to_ids = {v:k for k, v in additional_ids_to_tokens.items()}
try:
    ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, tokenizer)
except ValueError:
    print('Already updated')
print(additional_tokens_to_ids)


# Load model

import torch
from transformers import GPT2LMHeadModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
model.eval()
_ = model.to(device)

# Create context
def context_wrapper(masked_text):
	
	context = masked_text.strip()

	context_ids = ilm.tokenize_util.encode(context, tokenizer)

	# Replace blanks with appropriate tokens from left to right
	_blank_id = ilm.tokenize_util.encode(' _', tokenizer)[0]
	for i in range(1,len(context.split(' _'))):
  		context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_word|>']
	
	#context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_sentence|>']
	
	
	#print(ilm.tokenize_util.decode(context_ids, tokenizer))
	return(context_ids)

def ilm_generate(context_ids, additional_tokens_to_ids, num_iteration=2):
	
	generated = infill_with_ilm(
	    model,
	    additional_tokens_to_ids,
	    context_ids,
	    num_infills=num_iteration)
	generated_text = []
	for g in generated:
	    generated_text.append(ilm.tokenize_util.decode(g, tokenizer))
	return(generated_text)

context = """
non-Fiction
I want to your _, I want you _. I want you everything as long as it's _.
""".strip()	
context_ids = context_wrapper(context)
output = ilm_generate(context_ids, additional_tokens_to_ids,5)    
for item in output:
	print(item)