# Copyright (c) Meta Platforms, Inc. and affiliates
from pippy import annotate_split_points, PipeSplitWrapper


def add_split_points(roberta, encoders_per_rank):
    for i in range(0, roberta.config.num_hidden_layers // encoders_per_rank):
        annotate_split_points(
            roberta,
            {
                f"roberta.encoder.layer.{i}": PipeSplitWrapper.SplitPoint.BEGINNING
            },
        )
    annotate_split_points(
        roberta, {"lm_head": PipeSplitWrapper.SplitPoint.BEGINNING}
    )


def split(model, num_ranks):
    emb_head = 2  # embeddings + head
    num_of_ranks_for_encoders = num_ranks - emb_head
    encoders_per_rank = (
        model.config.num_hidden_layers + num_of_ranks_for_encoders - 1
    ) // num_of_ranks_for_encoders  # a divider of roberta.config.num_hidden_layers: [1, 2, 3, 4, 6, 12]
    # print(f"encoders_per_rank = {encoders_per_rank}")
    add_split_points(model, encoders_per_rank)
