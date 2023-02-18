# Copyright (c) Meta Platforms, Inc. and affiliates
from pippy import annotate_split_points, PipeSplitWrapper


def add_split_points(gpt2, decoders_per_rank):
    for i in range(0, gpt2.config.n_layer // decoders_per_rank):
        annotate_split_points(
            gpt2,
            {
                f"transformer.h.{i * decoders_per_rank}": PipeSplitWrapper.SplitPoint.BEGINNING
            },
        )
    annotate_split_points(
        gpt2, {"transformer.ln_f": PipeSplitWrapper.SplitPoint.BEGINNING}
    )


def split(model, num_ranks):
    emb_head = 2  # embeddings + head
    num_of_ranks_for_decoders = num_ranks - emb_head
    decoders_per_rank = (
        model.config.n_layer + num_of_ranks_for_decoders - 1
    ) // num_of_ranks_for_decoders  # a divider of model.config.n_layer: [1, 2, 3, 4, 6, 12]
    # print(f"encoders_per_rank = {decoders_per_rank}")
    add_split_points(model, decoders_per_rank)
