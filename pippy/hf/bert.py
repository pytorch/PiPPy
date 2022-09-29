# Copyright (c) Meta Platforms, Inc. and affiliates
from pippy import annotate_split_points, PipeSplitWrapper


def add_split_points(bert, encoders_per_rank):
    for i in range(0, bert.config.num_hidden_layers // encoders_per_rank):
        annotate_split_points(bert,
                              {f'bert.encoder.layer.{i * encoders_per_rank}': PipeSplitWrapper.SplitPoint.BEGINNING})
    annotate_split_points(bert, {'classifier': PipeSplitWrapper.SplitPoint.BEGINNING})


def wrap(model, training_args, pp_ranks):
    emb_head = 2  # embeddings + head
    master_emb_head = training_args.exclude_master + emb_head  # master + embeddings + head
    num_of_ranks_for_encoders = (len(pp_ranks) - master_emb_head)
    encoders_per_rank = (model.config.num_hidden_layers + num_of_ranks_for_encoders - 1) // num_of_ranks_for_encoders  # a divider of bert.config.num_hidden_layers: [1, 2, 3, 4, 6, 12]
    # print(f"encoders_per_rank = {encoders_per_rank}")
    number_of_workers = emb_head + model.config.num_hidden_layers // encoders_per_rank  # 3 + a divider of bert.config.num_hidden_layers: [4, 5, 6, 7, 9, 15]
    all_worker_ranks = pp_ranks[training_args.exclude_master:training_args.exclude_master + number_of_workers]
    # print(f"number_of_workers = {number_of_workers}")
    add_split_points(model, encoders_per_rank)
