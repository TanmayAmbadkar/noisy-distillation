import os
from torch.utils.tensorboard import SummaryWriter

class TBLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, step)
        
    def log_metrics(self, prefix, metrics_dict, step):
        for k, v in metrics_dict.items():
            self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def close(self):
        self.writer.close()
