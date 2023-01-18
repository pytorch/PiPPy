import os
from torch.distributed.checkpoint import SavePlan, FileSystemWriter


class MultiHostFileSystemWriter(FileSystemWriter):
    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        os.makedirs(self.path, exist_ok=True)
        return plan
