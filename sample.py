"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import TransformerConfig, EncoderDecoder
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 50 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
prompt = "malaysia 's financial markets are closed on monday for a public holiday ."  # "malaysia markets closed for holiday"
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    conf = TransformerConfig(**checkpoint['model_args'])
    model = EncoderDecoder(conf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
max_token_len = 250

if prompt == '':
    raise ValueError("Must enter a prompt with --prompt")

tokenized_prompt = tokenizer(text=prompt, return_tensors="pt")
prompt_token_len = tokenized_prompt['input_ids'].shape[1]
tokenized_output = tokenizer(text='', padding="max_length", max_length=prompt_token_len, return_tensors="pt")


for k,v in tokenized_prompt.items():
    tokenized_prompt[k] = torch.cat((tokenized_prompt[k], torch.zeros((1,max_new_tokens))), dim=1)

label_ids = torch.zeros_like(tokenized_prompt['input_ids'])
label_ids[0][0] = 101
label_attention_mask = torch.zeros_like(tokenized_prompt['input_ids'])
label_attention_mask[0][0] = 1
    
inputs = {
    'input_ids': tokenized_prompt['input_ids'].int(), 
    'input_attention_mask': tokenized_prompt['attention_mask'].bool(), 
    'label_ids': label_ids.int(),
    'label_attention_mask': label_attention_mask.bool()
}
print(inputs['label_ids'])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(inputs, max_new_tokens, temperature=temperature, top_k=top_k)
            
            print(tokenizer.decode(y))
            print('---------------')
