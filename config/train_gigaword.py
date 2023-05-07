
out_dir = 'out-gigaword'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'gigaword-summary'
wandb_run_name = 'encoder-decoder-summary'

dataset = 'gigaword'
gradient_accumulation_steps = 1
batch_size = 64

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1
n_vocab = 50257

learning_rate = 1e-4 
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.99 

warmup_iters = 100 # not super necessary potentially

device = 'cuda'
# on macbook also add
# device = 'cpu'  # run on cpu only
compile = True # currently only available on linux
