from src.evaluation.smoothness import SmoothnessEvaluator
from src.evaluation.robustness import RobustnessEvaluator
import torch

def evaluate_all(cfg, teacher, student, logger, env, idx=1):
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")
    
    # 1. Evaluate Smoothness
    evaluator = SmoothnessEvaluator(cfg, device)
    states = evaluator.collect_states(env, teacher)
    
    student_smoothness = evaluator.evaluate(student, states)
    robust_eval = RobustnessEvaluator(cfg, device)
    student_robustness = robust_eval.evaluate_noise_sweep(env, student, prefix=f"student_{idx}_robustness")
    
    metrics = {}
    
    if idx == 1:
        teacher_smoothness = evaluator.evaluate(teacher, states)
        teacher_robustness = robust_eval.evaluate_noise_sweep(env, teacher, prefix="teacher_robustness")
        
        for k, v in teacher_smoothness.items():
            metrics[f"teacher_{k}"] = v
        metrics.update(teacher_robustness)
        
        if logger is not None:
            for noise in robust_eval.noise_levels:
                logger.log_scalar("robustness_sweep/teacher", teacher_robustness[f"teacher_robustness/noise_{noise}_mean"], int(noise*1000))
                
    for k, v in student_smoothness.items():
        metrics[f"student_{idx}_{k}"] = v
        
    metrics.update(student_robustness)
        
    if logger is not None:
        for noise in robust_eval.noise_levels:
            logger.log_scalar(f"robustness_sweep/student_{idx}", student_robustness[f"student_{idx}_robustness/noise_{noise}_mean"], int(noise*1000))
            
    return metrics
