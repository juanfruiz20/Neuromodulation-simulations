def adv_weight_schedule(
    epoch: int,
    start_epoch: int = 1,
    ramp_end_epoch: int = 15,
    start_value: float = 1e-3,
    end_value: float = 1e-2
) -> float:
    """
    Linear ramp schedule for the adversarial loss weight.

    Before start_epoch:
        lambda_adv = 0

    Between start_epoch and ramp_end_epoch:
        lambda_adv increases linearly from start_value to end_value

    After ramp_end_epoch:
        lambda_adv = end_value
    """
    if epoch < start_epoch:
        return 0.0

    if epoch >= ramp_end_epoch:
        return float(end_value)

    alpha = float(epoch - start_epoch) / float(max(1, ramp_end_epoch - start_epoch))

    return float((1.0 - alpha) * start_value + alpha * end_value)