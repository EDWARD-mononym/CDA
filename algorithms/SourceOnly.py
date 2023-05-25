from algorithms.BaseModel import BaseModel

class SourceOnly(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)

        self.algo_name = "SourceOnly"

    def update(self, src_loader, trg_loader, timestep):
        pass
