"""
how tokenizer is defined:
1. the simplified drawing files (*.ndjson) stores drawings as positioned and scaled data into a 256x256 region; 
2. we quantize the 256x256 region into a 64x64 region: qn = n * 64 // 256; 
3. we encode the strokes information as one token per point: token = qy * 64 + qx;
4. therefore the strokes vocab is 4095=63*64+63; 
5. we add 3 special tokens: <|draw|>, <|stroke|>, <|endoftext|>  
6. therefore the <|endoftext|> is 4098.
7. and the whole vocab size is 4099
"""
_SIZE = 64
_STROKES_VOCAB = (_SIZE-1)*_SIZE + _SIZE-1
_SPECIAL_TOKENS = [
    '<|draw|>',
    '<|stroke|>',
    '<|endoftext|>'
] 
VOCAB_SIZE = _STROKES_VOCAB + len(_SPECIAL_TOKENS) + 1
PAD_ID = VOCAB_SIZE-1 # basically <|endoftext|> 

def q(n: int, max: int) -> int: 
    return n * _SIZE // max

def encode_point(point: tuple[int]) -> int: 
    qx, qy = point
    assert 0 <= qx <= _SIZE ; assert 0 <= qy <= _SIZE 
    return qy * _SIZE + qx

def encode_special(val: str) -> int:  
    return _STROKES_VOCAB + _SPECIAL_TOKENS.index(val) + 1

def encode(seq: list[tuple[int] | str]) -> list[int]: 
    tokens = []
    for el in seq: 
        if isinstance(el, tuple): enc = encode_point
        elif isinstance(el, str): enc = encode_special
        tokens.append(enc(el))
    return tokens 

def decode(tokens: list[int]) -> list[tuple[int] | str]: 
    seq = []
    for token in tokens: 
        if token <= _STROKES_VOCAB: 
            x = token % _SIZE
            y = token // _SIZE
            dec = (x,y)
        else: 
            dec = _SPECIAL_TOKENS[token - _STROKES_VOCAB - 1]
        seq.append(dec)
    return seq 

if __name__ == '__main__': 
    seq = ['<|draw|>', (0, 0), (63, 63), '<|stroke|>', (32, 0), (0, 32), '<|stroke|>', (5, 6), (7, 8), '<|stroke|>', '<|endoftext|>']
    seq_enc = encode(seq) 
    seq_dec = decode(seq_enc)
    print(f'size: {VOCAB_SIZE}')
    assert seq == seq_dec
    