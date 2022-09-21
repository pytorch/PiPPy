import torch
import torch.distributed
from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader


class PiPPyAccelerator(Accelerator):
    def __init__(self, args):
        if args.with_tracking:
            super().__init__(log_with=args.report_to, logging_dir=args.output_dir)
        else:
            super().__init__()
        self.args = args

    def wait_for_everyone(self):
        torch.distributed.barrier(self.args.driver_group)

    def prepare_model(self, model):
        self._models.append(model)
        return model

    def prepare_data_loader(self, data_loader):
        return prepare_data_loader(
            data_loader,
            self.device,
            num_processes=self.args.dp_group_size,
            process_index=self.process_index,
            split_batches=self.split_batches,
            put_on_device=self.device_placement,
            rng_types=self.rng_types.copy(),
            dispatch_batches=self.dispatch_batches,
        )
