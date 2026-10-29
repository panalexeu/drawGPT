import os 
import json
import warnings 
from fnmatch import fnmatch

import numpy as np 
from tqdm import tqdm
from google.cloud.storage.client import Client

from tokenizer import q, encode, PAD_ID

# fckn hate warnings!!!
warnings.filterwarnings(
    "ignore",
    message=".*end user credentials from Google Cloud SDK.*",
)

block_size = 256
# quickdraw gs 
bucket = 'quickdraw_dataset'
prefix = 'full/simplified/'
patterns = [
    "*line.ndjson",
    "*triangle.ndjson",
    "*zigzag.ndjson",
    "*leg.ndjson",
    "*square.ndjson",
    "*stairs.ndjson",
    "*canoe.ndjson",
    "*mountain.ndjson",
    "*pillow.ndjson",
    "*door.ndjson",
    "*swing set.ndjson",
    "*spoon.ndjson",
    "*fork.ndjson",
    "*ladder.ndjson",
    "*hexagon.ndjson",
    "*circle.ndjson",
    "*shovel.ndjson",
    "*pear.ndjson",
    "*lipstick.ndjson",
    "*wine bottle.ndjson",
    "*axe.ndjson",
    "*streetlight.ndjson",
] # *.ndjson
data_dir = './data/quickdraw/data/'
region_size = 256 # the size of simplified canvas is 256x256 region

def _list_files(client: Client): 
    blobs = client.list_blobs(bucket, prefix=prefix)
    return [blob.name for blob in blobs if any([fnmatch(blob.name, p) for p in patterns])]

def _download(client: Client) -> list[dict]:
    os.makedirs(data_dir, exist_ok=True)

    jsons = []
    for blob in tqdm(_list_files(client)):
        path = data_dir + blob.split('/')[-1]
        client.bucket(bucket).blob(blob).download_to_filename(path)
        jsons = jsons + _read_jsonl(path)
    return jsons

def _read_jsonl(path: str) -> list[dict]: 
    with open(path, 'r') as f: 
        lines = f.readlines()
    return [json.loads(line) for line in lines]

_word_idx = 0
_word_table = {}
def _get_word_id(word: str):
    global _word_idx, _word_table 
    try: 
        return _word_table[word]    
    except KeyError: 
        _word_table[word] = _word_idx
        _word_idx += 1
        return _word_table[word]

if __name__ == '__main__': 
    client = Client().create_anonymous_client()
    jsons = _download(client)

    arr = np.memmap(
        filename=data_dir + 'data.bin', 
        dtype=np.uint16, 
        mode='w+', 
        shape=(len(jsons), block_size)
    )
    for i, json_ in enumerate(jsons): 
        # assign a word to embedding table: 
        enc_word = [_get_word_id(json_['word'])]
        seq = ['<|draw|>']
        # json drawings:  [// First stroke [x0, x1, x2, x3, ...], [y0, y1, y2, y3, ...],],[// Second stroke[x0, x1, x2, x3, ...], [y0, y1, y2, y3, ...],], // Additional strokes ...
        for drawing in json_['drawing']:
            for x, y in zip(drawing[0], drawing[1]): 
                # quantize point
                qx, qy = q(x, region_size), q(y, region_size)
                seq.append((qx, qy))
            seq.append('<|stroke|>')
        seq.append('<|endoftext|>')
        enc_seq = encode(seq)
        enc_seq = enc_word + enc_seq
        enc_seq = np.array(enc_seq, dtype=np.uint16)
        # warn and cutoff if exceeds the block size
        if len(enc_seq) > block_size: 
            print(f'{i}: warning enc_seq len {len(enc_seq)} exceeds the specified block size: {block_size}')
            enc_seq = enc_seq[:block_size]
            enc_seq[-1] = PAD_ID
        else:
            enc_seq = np.pad(enc_seq, (0, block_size-len(enc_seq)), constant_values=PAD_ID) # silently cutoff if exceeds for now 
        arr[i, :] = enc_seq[:] 
    
    # save the word table 
    with open(data_dir + 'vocab.json', 'w') as f:
        json.dump(_word_table, f)
    