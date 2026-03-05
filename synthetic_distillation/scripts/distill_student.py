from src.distillation.distiller import Distiller
import torch

def distill(cfg, teacher, env, logger):
    from src.utils.device import get_device
    device = get_device(cfg.device)
    distiller = Distiller(cfg, device, logger)
    student = distiller.train(teacher, env)
    return student
