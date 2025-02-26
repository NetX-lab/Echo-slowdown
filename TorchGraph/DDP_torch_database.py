from torch_database import TorchDatabase

class DDPTorchDatabase(TorchDatabase):
    def __init__(self, module: torch.nn.Module, example: torch.tensor, name: str, timer: Timer, optimizer: Optimizer):
        super.__init__(module, example, name, timer, optimizer)