# Copyright (c) Meta Platforms, Inc. and affiliates
from pippy import annotate_split_points, PipeSplitWrapper


def add_split_points(bart, xxcoders_per_rank):
    assert bart.config.encoder_layers == bart.config.decoder_layers
    assert (
        bart.config.encoder_layers + bart.config.decoder_layers
    ) % xxcoders_per_rank == 0
    encoders_per_rank = xxcoders_per_rank
    for i in range(
        0,
        (bart.config.encoder_layers + encoders_per_rank - 1)
        // encoders_per_rank,
    ):
        annotate_split_points(
            bart,
            {
                f"model.encoder.layers.{i * encoders_per_rank}": PipeSplitWrapper.SplitPoint.BEGINNING
            },
        )
    decoders_per_rank = xxcoders_per_rank
    for i in range(
        0,
        (bart.config.decoder_layers + decoders_per_rank - 1)
        // decoders_per_rank,
    ):
        annotate_split_points(
            bart,
            {
                f"model.decoder.layers.{i * decoders_per_rank}": PipeSplitWrapper.SplitPoint.BEGINNING
            },
        )
    annotate_split_points(
        bart, {"lm_head": PipeSplitWrapper.SplitPoint.BEGINNING}
    )


def split(model, num_ranks):
    emb_head = 2  # encoder embeddings + decoder embeddings
    num_of_ranks_for_xxcoders = num_ranks - emb_head
    xxcoders = model.config.encoder_layers + model.config.decoder_layers
    xxcoders_per_rank = (
        xxcoders + num_of_ranks_for_xxcoders - 1
    ) // num_of_ranks_for_xxcoders
    # print(f"xxcoders_per_rank = {xxcoders_per_rank}")
    add_split_points(model, xxcoders_per_rank)
