import os
from transformers import T5ForConditionalGeneration, T5Config
from transformers import AutoModelForSeq2SeqLM

def add_split_points(t5, num_submodules):
    if num_submodules == 1:
        pass
    elif num_submodules == 3:
        # assert num_submodules == _add_split_points(t5, [16, 30])
        assert num_submodules == _add_split_points(t5, [17, 31])
    elif num_submodules == 4:
        assert num_submodules == _add_split_points(t5, [13, 24, 35])
    elif num_submodules == 7:
        # assert num_submodules == _add_split_points(t5, [8, 14, 20, 26, 32, 38])
        assert num_submodules == _add_split_points(t5, [9, 15, 21, 27, 33, 39])
    elif num_submodules == 8:
        # assert num_submodules == _add_split_points(t5, [7, 13, 19, 25, 31, 37, 43])
        assert num_submodules == _add_split_points(t5, [9, 14, 19, 24, 29, 34, 39])
    elif num_submodules == 15:
        # assert num_submodules == _add_split_points(t5, [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42])
        assert num_submodules == _add_split_points(t5, [1, 5, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42])
    elif num_submodules == 16:
        # assert num_submodules == _add_split_points(t5, [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 44])
        # assert num_submodules == _add_split_points(t5, [1, 4, 7, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41])
        assert num_submodules == _add_split_points(t5, [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43])
    else:
        raise ValueError(f'Unsupported num_submodules = {num_submodules}')


def _add_split_points(t5, split_indices):
    enc_emb = 1
    num_enc = t5.config.num_layers
    dec_emb = 1
    num_dec = t5.config.num_decoder_layers
    lm_head = 1
    count = 0
    for index in split_indices:
        if index < enc_emb:
            # index = 0: do nothing
            pass
        elif index < enc_emb + num_enc:
            if index == enc_emb:
                # index = 1: insert a split point after `encoder.embed_tokens` before the first encoder
                # to put encoder's dropout with the first encoder and not with encoders' embeddings
                annotate_split_points(t5, {f'encoder.embed_tokens': PipeSplitWrapper.SplitPoint.END})
            else:
                # 1 < index < 1 + num_enc: insert a split point before the `index - enc_emb`-th encoder
                annotate_split_points(t5, {f'encoder.block.{index - enc_emb}': PipeSplitWrapper.SplitPoint.BEGINNING})
            count += 1
        elif index < enc_emb + num_enc + dec_emb + num_dec:
            # 1 + num_enc <= index < 1 + num_enc + 1 + num_dec
            if index == enc_emb + num_enc:
                # index = 1 + num_enc: insert a split point before `decoder.embed_tokens`
                annotate_split_points(t5, {f'decoder.embed_tokens': PipeSplitWrapper.SplitPoint.BEGINNING})
            elif index == enc_emb + num_enc + dec_emb:
                # index = 1 + num_enc + 1: insert a split point after `decoder.embed_tokens` before the first decoder
                # to put decoder's dropout with the first decoder and not with decoders' embeddings
                annotate_split_points(t5, {f'decoder.embed_tokens': PipeSplitWrapper.SplitPoint.END})
            else:
                # 1 + num_enc + 1 < index < 1 + num_enc + 1 + num_dec:
                # insert a split point before the `index - (enc_emb + num_enc + dec_emb)`-th encoder
                annotate_split_points(t5, {
                    f'decoder.block.{index - (enc_emb + num_enc + dec_emb)}': PipeSplitWrapper.SplitPoint.BEGINNING})
            count += 1
        elif index < enc_emb + num_enc + dec_emb + num_dec + lm_head:
            # index = 1 + num_enc + 1 + num_dec: insert a split point before the `lm_head`
            annotate_split_points(t5, {f'lm_head': PipeSplitWrapper.SplitPoint.BEGINNING})
            count += 1
    return count + 1
