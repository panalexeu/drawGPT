import os 
import time 
import json
from contextlib import nullcontext

import torch
from PIL import Image, ImageDraw

from model import GPTConfig, GPT
from tokenizer import PAD_ID, decode, SPECIAL_TOKENS

seed = 2004 
size = 64
width = 3 
max_new_tokens = 256
temperature = 1.0
top_k = 64
stop_tokens = [PAD_ID]
word_vocab_path = './data/quickdraw/vocab.json'
word_vocab = json.load(open(word_vocab_path, 'r'))
ckpt_path = './out/ckpt.pt'
device = 'cuda'
dtype = 'bfloat16'
compile = False

def _draw(tokens: list[int], save_fig: bool = False, out_dir: str = './doodles'):
    tokens = decode(tokens)
    img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(img)

    frames = []
    prev = None
    for token in tokens:
        if token == '<|endoftext|>':
            break
        if token in SPECIAL_TOKENS:          # <|stroke|> / <|draw|>: lift the pen
            prev = None
            continue
        if prev is not None:
            draw.line([prev, token], fill=255, width=width)
            if save_fig:
                frames.append(img.copy())
        prev = token

    if save_fig and frames:
        os.makedirs(out_dir, exist_ok=True)
        name = time.strftime('%Y%m%d-%H%M%S')
        durations = [40] * (len(frames) - 1) + [1500]   # hold the finished drawing
        frames[0].save(os.path.join(out_dir, f'{name}.gif'), save_all=True,
                       append_images=frames[1:], duration=durations, loop=0)

    return img

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
            img = _draw(idx.squeeze(0).tolist(), save_fig=True)
            img.show()