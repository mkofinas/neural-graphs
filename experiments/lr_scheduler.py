import math

from torch.optim.lr_scheduler import LRScheduler


class WarmupLRScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps=10000, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [
                base_lr * self._step_count / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            return self.base_lrs


class ExpLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps=10000,
        decay_rate=0.5,
        decay_steps=100000,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            learning_rates = [
                base_lr * self._step_count / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            learning_rates = self.base_lrs
        learning_rates = [
            lr * (self.decay_rate ** (self._step_count / self.decay_steps))
            for lr in learning_rates
        ]
        # print(self._step_count, learning_rates)
        return learning_rates


class CosLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        decay_steps,
        last_epoch=-1,
        alpha=0.0,
        verbose=False,
    ):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            learning_rates = [
                base_lr * self._step_count / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            decay_steps = self.decay_steps - self.warmup_steps
            step = min(self._step_count - self.warmup_steps, decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.decay_steps))
            decayed = (1 - self.alpha) * cosine_decay + self.alpha
            learning_rates = [lr * decayed for lr in self.base_lrs]
        return learning_rates
