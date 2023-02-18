# Copyright (c) Meta Platforms, Inc. and affiliates
from pippy import annotate_split_points, PipeSplitWrapper


def add_split_points(bert, encoders_per_rank):
    for i in range(0, bert.config.num_hidden_layers // encoders_per_rank):
        annotate_split_points(
            bert,
            {
                f"bert.encoder.layer.{i * encoders_per_rank}": PipeSplitWrapper.SplitPoint.BEGINNING
            },
        )
    annotate_split_points(
        bert, {"classifier": PipeSplitWrapper.SplitPoint.BEGINNING}
    )


def split(model, num_ranks):
    emb_head = 2  # embeddings + head
    num_of_ranks_for_encoders = num_ranks - emb_head
    encoders_per_rank = (
        model.config.num_hidden_layers + num_of_ranks_for_encoders - 1
    ) // num_of_ranks_for_encoders  # a divider of bert.config.num_hidden_layers: [1, 2, 3, 4, 6, 12]
    # print(f"encoders_per_rank = {encoders_per_rank}")
    add_split_points(model, encoders_per_rank)
