from src.evaluation.smoothness import SmoothnessEvaluator
from src.evaluation.robustness import RobustnessEvaluator
import torch

def evaluate_all(cfg, teacher, student, logger, env, idx=1):
    from src.utils.device import get_device
    device = get_device(cfg.device)
    
    # 1. Evaluate Smoothness
    evaluator = SmoothnessEvaluator(cfg, device)
    real_states = evaluator.collect_states(env, teacher)
    
    # Generate gaussian states
    std = 0.05
    if "sampling" in cfg.distill and "std" in cfg.distill.sampling:
        std = cfg.distill.sampling.std
    gaussian_noise = torch.randn_like(real_states) * std
    gaussian_states = real_states + gaussian_noise
    
    student_noise_prefix = getattr(student, "metadata", {}).get("noise_name", "")
    student_metric_prefix = f"student_{idx}_{student_noise_prefix}" if student_noise_prefix else f"student_{idx}"
    
    student_smoothness = evaluator.evaluate(student, real_states, gaussian_states)
    robust_eval = RobustnessEvaluator(cfg, device)
    student_robustness = robust_eval.evaluate_noise_sweep(env, student, prefix=f"{student_metric_prefix}_robustness")
    
    metrics = {}
    
    if idx == 1:
        teacher_smoothness = evaluator.evaluate(teacher, real_states, gaussian_states)
        teacher_robustness = robust_eval.evaluate_noise_sweep(env, teacher, prefix="teacher_robustness")
        
        for k, v in teacher_smoothness.items():
            metrics[f"teacher_{k}"] = v
        metrics.update(teacher_robustness)
        
        if logger is not None:
            for noise in robust_eval.noise_levels:
                logger.log_scalar("robustness_sweep/teacher", teacher_robustness[f"teacher_robustness/noise_{noise}_mean"], int(noise*1000))
                
    for k, v in student_smoothness.items():
        metrics[f"{student_metric_prefix}_{k}"] = v
        
    metrics.update(student_robustness)
        
    if logger is not None:
        for noise in robust_eval.noise_levels:
            logger.log_scalar(f"robustness_sweep/{student_metric_prefix}", student_robustness[f"{student_metric_prefix}_robustness/noise_{noise}_mean"], int(noise*1000))
            
    return metrics
