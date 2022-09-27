# Copyright (c) Meta Platforms, Inc. and affiliates
import os
import torch
from spmd.tensor import DTensor, DeviceMesh, Shard, Replicate


def synthetic_data(w, b, num_examples):
    X = torch.randn(num_examples, len(w))
    y = torch.matmul(X, w.reshape((2, 1)).to(X.device)) + b
    y += torch.randn(*y.shape) * 0.01
    return X, y.reshape((-1, 1))


def model(X, w, b):
    return torch.addmm(b, X, w)


def loss_func(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(
        backend="nccl", world_size=world_size, rank=rank
    )

    mesh = DeviceMesh("cuda", list(range(world_size)))
    shard_0_placement = [Shard[0]]
    replica_placement = [Replicate()]

    # params of the ground truth model
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    print(f"rank = {rank}, true_w = {true_w}, true_b = {true_b}")

    # model params, replicated to all ranks with `placements=[Replicate()]`
    w = spmd.DTensor(
        [[100.0], [100.0]],
        requires_grad=True,
        device_mesh=mesh,
        placements=replica_placement,
    )
    b = spmd.zeros(
        1, requires_grad=True, device_mesh=mesh, placements=replica_placement
    )
    print(f"rank = {rank}, w = {w}, b = {b}")

    optimizer = torch.optim.SGD([w, b], lr=1e-3)
    num_epochs, num_iters, batch_size = 5, 100, 10
    for epoch in range(num_epochs):
        for i in range(num_iters):
            # X, y are local tensors, we need to create distributed tensor from local
            # torch.Tensor, and use DTensor as model input to implement data parallel
            X, y = synthetic_data(true_w, true_b, batch_size)
            g_X = DTensor.from_local(
                x, device_mesh=mesh, placements=shard_0_placement
            )
            g_y = DTensor.from_local(
                y, device_mesh=mesh, placements=shard_0_placement
            )
            l = loss_func(model(g_X, w, b), g_y)

            optimizer.zero_grad()
            l.sum().backward()
            optimizer.step()

    print(f"rank = {rank}, w = {w}, b = {b}")


if __name__ == "__main__":
    main()
