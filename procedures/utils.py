import torch

def apply_non_binary_threshold(task_vector, threshold):

    task_vector_abs = task_vector.abs().cpu()
    torch.cuda.empty_cache()
    pos = task_vector_abs.topk(int(threshold * task_vector_abs.numel())).indices

    return task_vector, pos