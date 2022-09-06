from collections import Iterator

import torch
from hypergrad import stoch_AID


class BiOptimizer:
    def inner_step(self, inner_loss):
        raise NotImplementedError

    def outer_step(self, outer_loss, fp_map=None, inner_loss=None):
        raise NotImplementedError


class TorchBiOptimizer(BiOptimizer):
    def __init__(
        self,
        inner_parameters,
        outer_parameters,
        inner_iterator,
        outer_iterator,
        reverse_iterator,
        inner_batch_size,
        outer_batch_size,
        T=10,
        K=10,
        J=1,
        inner_lr=0.1,
        outer_lr=0.1,
        linsys_lr=None,
        inner_optimizer=torch.optim.SGD,
        outer_optimizer=torch.optim.SGD,
        linsys_optimizer=torch.optim.SGD,
        inner_optimizer_kwargs={},
        outer_optimizer_kwargs={},
        linsys_optimizer_kwargs={},
        warm_start=False,
        warm_start_linsys=False,
        stoch_inner=False,
        stoch_outer=False,
        after_hypergrad_callback=None,
    ):

        self.outer_step_count = 0
        self.inner_step_count = 0
        self.inner_batch_size = inner_batch_size
        self.outer_batch_size = outer_batch_size

        self.T = (lambda: T(self.outer_step_count)) if callable(T) else lambda: T
        self.K = (lambda: K(self.outer_step_count)) if callable(K) else lambda: K
        self.J = (lambda: J(self.outer_step_count)) if callable(J) else lambda: J
        self.inner_lr = (
            (lambda: inner_lr(self.outer_step_count, self.inner_step_count))
            if callable(inner_lr)
            else lambda: inner_lr
        )
        self.outer_lr = (
            (lambda: outer_lr(self.outer_step_count))
            if callable(outer_lr)
            else lambda: outer_lr
        )

        if linsys_lr is None:
            self.linsys_lr = (
                (lambda x: inner_lr(self.outer_step_count, x))
                if callable(inner_lr)
                else lambda x: inner_lr
            )
        else:
            self.linsys_lr = (
                (lambda x: linsys_lr(self.outer_step_count, x))
                if callable(linsys_lr)
                else lambda x: linsys_lr
            )

        class Scheduler:
            linsys_lr = self.linsys_lr
            first_lr = self.linsys_lr(1)

            def __init__(self, optimizer):
                self.count = 2
                self.optimizer = optimizer

            def step(self):
                lr = Scheduler.linsys_lr(self.count)
                TorchBiOptimizer.set_lr(self.optimizer, lr)
                self.count += 1
                return lr

        self.Scheduler = Scheduler

        class CountedIterator(Iterator):
            def __init__(self, it):
                self.it = it
                self.calls = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.calls += 1
                return self.it.__next__()

        self.inner_iterator = CountedIterator(inner_iterator)
        self.outer_iterator = CountedIterator(outer_iterator)
        self.reverse_iterator = CountedIterator(reverse_iterator)

        self.inner_optimizer_class = inner_optimizer
        self.inner_optimizer_kwargs = inner_optimizer_kwargs
        self.inner_optimizer = None
        self.outer_optimizer: torch.optim.Optimizer = outer_optimizer(
            outer_parameters, lr=self.outer_lr(), **outer_optimizer_kwargs
        )

        self.linsys_optimizer_f = linsys_optimizer
        self.linsys_optimizer_kwargs = linsys_optimizer_kwargs

        self.warm_start = warm_start
        self.warm_start_linsys = warm_start_linsys
        self.stoch_inner = stoch_inner
        self.stoch_outer = stoch_outer
        self.after_hypergrad_callback = after_hypergrad_callback

        self.J_inner = self.J if self.stoch_inner else lambda: 1
        self.J_outer = self.J if self.stoch_outer else lambda: 1

        self.outer_parameters = outer_parameters
        self.inner_parameters = inner_parameters

        self.inner_parameters_start = [p.detach().clone() for p in inner_parameters]
        self.linsys_parameters = [torch.zeros_like(p) for p in inner_parameters]

    def get_budget(self):
        return (
            (self.inner_iterator.calls + self.reverse_iterator.calls)
            * self.inner_batch_size
            + self.outer_iterator.calls * self.outer_batch_size
        )

    def linsys_optimizer_build(self, parameters, **kwargs):
        opt = self.linsys_optimizer_f(parameters, lr=self.Scheduler.first_lr, **kwargs)
        scheduler = self.Scheduler(opt)
        return opt, scheduler

    @staticmethod
    def set_lr(optimizer, new_lr):
        if isinstance(optimizer, torch.optim.SGD):
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr

    @staticmethod
    def set_gradients(parameters, gradients):
        for p, g in zip(parameters, gradients):
            p.grad = g

    def outer_step(self, outer_loss, fp_map=None, inner_loss=None):
        assert self.inner_step_count == self.T(), (
            f"wrong call to outer_step, not enough ({self.T()}) inner_steps\n"
            f"outer_step={self.outer_step_count}, inner_step={self.inner_step_count}"
        )
        assert (fp_map is not None) != (inner_loss is not None)

        self.outer_optimizer.zero_grad()
        self.inner_optimizer.zero_grad()

        linsys_start = self.linsys_parameters if self.warm_start_linsys else None

        def outer_loss_noiter(params, hparams):
            return outer_loss(params, hparams, next(self.outer_iterator))

        if fp_map is not None:
            inner_loss_noiter = None

            def fp_map_noiter(params, hparams):
                return fp_map(params, hparams, next(self.reverse_iterator))

        else:
            fp_map_noiter = None

            def inner_loss_noiter(params, hparams):
                return inner_loss(params, hparams, next(self.reverse_iterator))

        for K in (self.K(), 0):
            try:
                _, self.linsys_parameters = self.get_hypergrad(
                    params=self.inner_parameters,
                    hparams=self.outer_parameters,
                    outer_loss=outer_loss_noiter,
                    K=K,
                    J_inner=self.J_inner(),
                    J_outer=self.J_outer(),
                    fp_map=fp_map_noiter,
                    inner_loss=inner_loss_noiter,
                    linsys_start=linsys_start,
                    stoch_outer=self.stoch_outer,
                    stoch_inner=self.stoch_inner,
                    optim_build=self.linsys_optimizer_build,
                    opt_params=self.linsys_optimizer_kwargs,
                )
                break
            except ValueError as e:
                print(e)
                if K == 0:
                    raise e

        if self.after_hypergrad_callback is not None:
            self.after_hypergrad_callback(self.outer_parameters)

        self.outer_optimizer.step()

        self.outer_step_count += 1
        self.set_lr(self.outer_optimizer, self.outer_lr())

    def get_hypergrad(self, *args, **kwargs):
        return stoch_AID(*args, **kwargs)

    def init_inner(self):
        self.inner_step_count = 0

        if not self.warm_start:
            for p, p_s in zip(self.inner_parameters, self.inner_parameters_start):
                p.data = p_s.detach().clone()

        if (not self.warm_start) or self.outer_step_count == 0:
            self.inner_optimizer = self.inner_optimizer_class(
                self.inner_parameters, lr=self.inner_lr(), **self.inner_optimizer_kwargs
            )

    def inner_step(self, inner_loss):
        assert self.inner_step_count < self.T(), (
            f"wrong call to inner_step, exceeding the allowed {self.T()}\n"
            f"outer_step={self.outer_step_count}, inner_step={self.inner_step_count}"
        )

        t_loss = inner_loss(
            self.inner_parameters, self.outer_parameters, next(self.inner_iterator)
        )
        t_loss.backward(inputs=self.inner_parameters)

        self.inner_optimizer.step()
        self.inner_optimizer.zero_grad()

        self.inner_step_count += 1

        self.set_lr(self.inner_optimizer, self.inner_lr())

        return t_loss
