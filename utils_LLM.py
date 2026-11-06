import numpy as np


def rampup(global_step: int, rampup_length: int = 100) -> float:
    if global_step < rampup_length:
        global_step = float(global_step)
        rampup_length = float(rampup_length)
        phase = 1.0 - np.maximum(0.0, global_step) / rampup_length
    else:
        phase = 0.0
    return float(np.exp(-5.0 * phase * phase))
