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


def wrap(model, training_args, pp_ranks):
    emb_head = 2  # embeddings + head
    master_emb_head = (
        training_args.exclude_master + emb_head
    )  # master + embeddings + head
    num_of_ranks_for_decoders = len(pp_ranks) - master_emb_head
    decoders_per_rank = (
        model.config.n_layer + num_of_ranks_for_decoders - 1
    ) // num_of_ranks_for_decoders  # a divider of model.config.n_layer: [1, 2, 3, 4, 6, 12]
    # print(f"encoders_per_rank = {decoders_per_rank}")
    number_of_workers = (
        emb_head + model.config.n_layer // decoders_per_rank
    )  # 3 + a divider of model.config.n_layer: [4, 5, 6, 7, 9, 15]
    all_worker_ranks = pp_ranks[
        training_args.exclude_master : training_args.exclude_master
        + number_of_workers
    ]
    # print(f"number_of_workers = {decoders_per_rank}")
    add_split_points(model, decoders_per_rank)
