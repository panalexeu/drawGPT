import json
from contextlib import nullcontext

import torch
from PIL import Image, ImageDraw

from model import GPTConfig, GPT
from tokenizer import PAD_ID, decode, SPECIAL_TOKENS

seed = 2004 
size = 64
width = 3 
max_new_tokens = 512
temperature = 1.0 
top_k = None
stop_tokens = [PAD_ID]
word_vocab_path = './data/quickdraw/vocab.json'
word_vocab = json.load(open(word_vocab_path, 'r'))
ckpt_path = './out/ckpt.pt'
device = 'cuda'
dtype = 'bfloat16'
compile = False

def _draw(tokens: list[int]):
    tokens = decode(tokens)
    img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(img)

    prev = None
    for token in tokens:
        if token in  SPECIAL_TOKENS: 
            continue 
        if prev is not None: 
            draw.line([prev, token], fill=255, width=width)
        prev = token
    img.show()

if __name__ == '__main__': 
    # torch stuff init 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # load and init model 
    ckpt = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**ckpt['model_args'])
    model = GPT(gptconf)
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if compile: model = torch.compile(model)

    while True: 
        prompt = input('input: ') 
        prompt_enc = [word_vocab[prompt]]
        x = torch.tensor(prompt_enc, dtype=torch.long, device=device).unsqueeze(0) # add batch dim

        with ctx: 
            gen_iter = model.gen_iter(x, max_new_tokens, stop_tokens, temperature, top_k)
            for i, idx in enumerate(gen_iter, start=1):
                token = idx.squeeze(0).tolist()[i] # squeeze batch back
                print(*decode(token), end='', flush=True)
            print()
            _draw(idx.squeeze(0).tolist())
