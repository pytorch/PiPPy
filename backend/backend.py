import torch
import torch.fx

class PipelineStageExecutor:
    def execute(self, forward_code, loss_code, schedule):
        # TODO: pass descriptor for value connectivity from other stages
        for µbatch_id, phase in schedule:
            if phase == 'fl':
                import pdb; pdb.set_trace()
                pass
            elif phase == 'f':
                raise NotImplementedError()
            elif phase == 'b':
                raise NotImplementedError()
            else:
                raise RuntimeError(f'Unknown phase {phase}')

class MyTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(512, 512)

    def forward(self, x):
        return self.lin(x)

traced = torch.fx.symbolic_trace(MyTestModel())

def loss(x):
    return torch.sum(x)

traced_loss = torch.fx.symbolic_trace(loss)

pse = PipelineStageExecutor()

schedule = []
chunks = 8
for i in range(chunks):
    schedule.extend([
        (i, 'fl'),
        (i, 'b')
    ])

pse.execute(traced, loss, schedule)