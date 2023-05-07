from dataclasses import dataclass
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import inspect
from torch.nn.utils.rnn import pad_sequence


@dataclass
class TransformerConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    device: str = 'cpu'


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class PositionalEncoding(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd, device=config.device)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embd, device=config.device)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_idx):
        b, seq_len = token_idx.size()

        pos = torch.arange(0, seq_len, dtype=torch.long, device=token_idx.device).unsqueeze(0)
        self.tok_emb = self.token_embeddings(token_idx)
        self.pos_emb = self.position_embeddings(pos)

        return self.dropout(self.tok_emb + self.pos_emb)


class SelfAttention(nn.Module):

    def __init__(self, config, causal):
        super().__init__()
        # Ensure that the embedding can be split up into n heads
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.causal = causal

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # 3 * config.n_embd so that the output can be split into key, query and value tensors.
        # Saves having to make 3 different linear layers
        self.qvk_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x, enc_in=None, key_padding_mask=None):
        b, seq_len, n_embd = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.qvk_proj(x).split(self.n_embd, dim=2)

        if enc_in is not None:
            _, k, v  = self.qvk_proj(enc_in).split(self.n_embd, dim=2)

        q = q.view(b, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(b, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(b, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(b, 1, 1, seq_len).expand(-1, self.n_head, -1, -1).reshape(b * self.n_head, 1, seq_len)
            attn_mask = key_padding_mask.view(b, self.n_head, -1, seq_len)
        else:
            attn_mask = None

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # much faster than other implementation!!
            attn_weight = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        else:
            attn_dropout = self.dropout if self.training else 0
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            # Mask padding tokens
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask == 0, -float('inf')) # Mask out pad tokens
            # Mask future tokens
            if self.causal:
                attn_mask = torch.ones(seq_len, seq_len, device=x.device).tril(diagonal=0)
                attn = attn.masked_fill(attn_mask == 0, -float('inf')) # Mask out future tokens
            attn_weight = torch.softmax(attn, dim=-1)
            attn_weight = torch.dropout(attn_weight, attn_dropout, self.training) 
            attn_weight  = attn_weight @ v # (b, nh, seq_len, seq_len) x (b, nh, seq_len, hs) -> (b, nh, seq_len, hs)
            
        attn_weight = attn_weight.transpose(1, 2).contiguous().view(b, seq_len, n_embd) # re-assemble all head outputs side by side
        # output projection
        attn_weight = self.resid_dropout(self.out_proj(attn_weight))
        
        return attn_weight


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.embd_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.embd_proj(x)
        x = self.dropout(x)
        return x

    
class EncoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.multi_head_attn = SelfAttention(config, causal=False)
        self.layer_norm_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.feed_forward = MLP(config)
        self.layer_norm_2 = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x, key_padding_mask=None):
        x = x + self.multi_head_attn(self.layer_norm_1(x), key_padding_mask=key_padding_mask)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)])
        self.layer_norm = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return self.layer_norm(x)
    
class DecoderBlock(nn.Module):

    def __init__(self, config, take_encoder_input):
        super().__init__()

        self.masked_multi_head_attn = SelfAttention(config, causal=True)
        self.layer_norm_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.feed_forward = MLP(config)
        self.layer_norm_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.take_encoder_input = take_encoder_input

        # Check if the decoder will take input from an encoder
        if take_encoder_input:
            self.multi_head_attn = SelfAttention(config, causal=False)
            self.layer_norm_2 = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x, enc_in=None, key_padding_mask=None):
        x = x + self.masked_multi_head_attn(self.layer_norm_1(x), key_padding_mask=key_padding_mask)
        if self.take_encoder_input:
            x = x + self.multi_head_attn(self.layer_norm_2(x), enc_in=enc_in, key_padding_mask=key_padding_mask)
        x = x + self.feed_forward(self.layer_norm_3(x))
        return x

class Decoder(nn.Module):

    def __init__(self, config, take_encoder_input=False):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([DecoderBlock(config, take_encoder_input=take_encoder_input) for _ in range(config.n_layer)])
        self.layer_norm = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x, enc_in, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, enc_in=enc_in, key_padding_mask=key_padding_mask)
        return self.layer_norm(x)

class EncoderDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.device = config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.transformer = nn.ModuleDict(dict(
            positional_enc = PositionalEncoding(config),
            encoder = Encoder(config),
            decoder = Decoder(config, True),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.positional_enc.token_embeddings.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
       
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('embd_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def forward(self, inputs):
        # shift labels by replacing token with padding token 
        target_true = torch.cat((inputs['label_ids'][:, 1:], torch.tensor([0]).expand(len(inputs['input_ids']),1)), dim=1).to(self.device)
        inputs['label_ids'][:, inputs['label_ids'].argmin()-1] = 0

        # Positional encoding
        in_pos_enc = self.transformer.positional_enc(inputs['input_ids'].to(self.device))
        
        # Encoder
        enc_attention_scores = self.transformer.encoder(in_pos_enc, key_padding_mask=inputs['input_attention_mask'].bool().to(self.device))
        target_pos_enc = self.transformer.positional_enc(inputs['label_ids'].to(self.device))

        # Decoder
        decoder_out = self.transformer.decoder(target_pos_enc, enc_in=enc_attention_scores)

        # if we are given some desired targets also calculate the loss
        logits = self.lm_head(decoder_out)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_true.contiguous().view(-1), ignore_index=0)

        return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer
    
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.positional_enc.position_embeddings.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(self, inputs, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        eos_token = 101
        output = []
        
        for i in range(1, max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(inputs)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Stop if eos token
            if idx_next.item() == eos_token:
                break
            # append sampled index to the running sequence and continue
            inputs['label_ids'][0][i] = idx_next
            inputs['label_attention_mask'][0][i] = True
            output.append(idx_next.item())

        return output