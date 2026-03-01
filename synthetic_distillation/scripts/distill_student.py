from src.distillation.distiller import Distiller
import torch

def distill(cfg, teacher, env, logger):
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")
    distiller = Distiller(cfg, device, logger)
    student = distiller.train(teacher, env)
    return student
