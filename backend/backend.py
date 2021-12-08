import torch
import torch.fx

from typing import List, NamedTuple

class PipelineStageExecutor:
    def __init__(self, forward_code, loss_code, input_connectivity, output_connectivity, schedule):
        self.forward_code = forward_code
        self.loss_code = loss_code
        self.input_connectivity = input_connectivity
        self.output_connectivity = output_connectivity
        self.schedule = schedule

    def execute(self, ):
        # TODO: pass descriptor for value connectivity from other stages
        for µbatch_idx, phase in self.schedule:
            if phase in {'fl', 'f'}:
                # TODO: args && kwargs?
                # TODO: more expressive types?
                args = self._recv_activations(µbatch_idx)

            if phase in {'b'}:
                # TODO: args && kwargs?
                # TODO: more expressive types?
                args = self._recv_gradients(µbatch_idx)

            if phase == 'fl':
                import pdb; pdb.set_trace()
                pass
            elif phase == 'f':
                raise NotImplementedError()
            elif phase == 'b':
                raise NotImplementedError()
            else:
                raise RuntimeError(f'Unknown phase {phase}')

    def _recv_activations(self, µbatch_idx):
        pass

    def _recv_gradients(self, µbatch_idx):
        pass


# This is the orginal module that we are going to pipeline
# We are going to pipeline it manually here to decouple development of
# the frontend from the backend
class SampleModuleToPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(512, 512)
        self.lin2 = torch.nn.Linear(512, 512)
        self.lin3 = torch.nn.Linear(512, 512)

    def forward(self, x):
        x = self.lin1(x)
        skip = x
        x = self.lin2(x)
        x = self.lin3(x)
        x = x + skip
        return x


class SampleModuleToPipelineStage0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(512, 512)

    def forward(self, x):
        return self.lin1(x)

class SampleModuleToPipelineStage1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin3 = torch.nn.Linear(512, 512)

    def forward(self, x, skip):
        x = self.lin3(x)
        return x + skip

class SampleModuleToPipelineStage2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin2 = torch.nn.Linear(512, 512)

    def forward(self, x):
        return self.lin2(x)

stage_modules = [
    torch.fx.symbolic_trace(SampleModuleToPipelineStage0()),
    torch.fx.symbolic_trace(SampleModuleToPipelineStage1()),
    torch.fx.symbolic_trace(SampleModuleToPipelineStage2()),
] 

class Connectivity(NamedTuple):
    input_connections : List[int]
    output_connections : List[int]

stage_connectivity = [
    
]

def loss(x):
    return torch.sum(x)

traced_loss = torch.fx.symbolic_trace(loss)

def fill_drain_schedule(chunks, loss=False):
    schedule = []
    schedule.extend((i, 'fl' if loss else 'f') for i in range(chunks))
    schedule.extend((i, 'b') for i in range(chunks))
    return schedule

# TODO
CHUNKS = 8

for i, mod in enumerate(stage_modules):
    is_final_module = i == len(stage_modules) - 1

    schedule = fill_drain_schedule(CHUNKS, is_final_module)

    pse = PipelineStageExecutor(mod, traced_loss if is_final_module else None, )

    import pdb; pdb.set_trace()