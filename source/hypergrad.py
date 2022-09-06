import torch
from torch.autograd import grad as torch_grad
from torch import Tensor
from torch.optim import Optimizer
from typing import List, Callable, Union, Any, Tuple


def stoch_AID(
    params: List[Tensor],
    hparams: List[Tensor],
    outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
    K: int,
    J_inner: int = 1,
    J_outer: int = 1,
    fp_map: Union[Callable[[List[Tensor], List[Tensor]], List[Tensor]], None] = None,
    inner_loss: Union[Callable[[List[Tensor], List[Tensor]], Tensor], None] = None,
    linsys_start: Union[List[Tensor], None] = None,
    stoch_outer: bool = False,
    stoch_inner: bool = False,
    optim_build: Union[Callable[..., Tuple[Optimizer, Any]], None] = None,
    opt_params: dict = None,
    set_grad: bool = True,
    verbose: bool = True,
):
    """
    Computes the hypergradient by solving the linear system by applying K steps of the optimizer output of the optim_build function,
    this should be a torch.optim.Optimizer. optim_build (optionally) returns also a scheduler whose step()
    method is called after every iteration of the optimizer.

    Args:
        params: the output of the inner solver procedure.
        hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
        outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
        K: the number of iteration of the LS solver which is given as output of optim_build
        J_inner: the minibatch size used to compute the jacobian w.r.t. hparams of fp_map
        J_outer: the minibatch size used to compute the gradient w.r.t. params and hparams  of outer_loss
        fp_map: the fixed point map which defines the inner problem, used if inner_loss is None
        inner_loss: the loss of the inner problem, used if fp_map is None
        linsys_start: starting point of the linear system, set to the 0 vector if None
        stoch_outer: set to True if outer_loss is stochastic, otherwise set to False
        stoch_inner: set to True if fp_map or inner_loss is stochastic, otherwise False
        optim_build: function used to obtain the linear system optimizer
        opt_params: parameters of he linear system optimizer (input of optim_build)
        set_grad: if True set t.grad to the hypergradient for every t in hparams
        verbose: print the distance between two consecutive iterates for the linear system.
    Returns:
        the list of hypergradients for each element in hparams
    """

    assert stoch_inner or (J_inner == 1)
    assert stoch_outer or (J_outer == 1)

    params = [w.detach().clone().requires_grad_(True) for w in params]

    if fp_map is not None:
        w_update_f = fp_map

        def v_update(v, jtv, g):
            return v - jtv - g

    elif inner_loss is not None:
        # In this case  w_update_f is the negative gradient
        def w_update_f(params, hparams):
            return torch.autograd.grad(
                -inner_loss(params, hparams), params, create_graph=True
            )

        def v_update(v, jtv, g):
            return -jtv - g

    else:
        raise NotImplementedError("Either fp_map or inner loss should be not None")

    o_loss = outer_loss(params, hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(
        o_loss, params, hparams, retain_graph=False
    )
    if stoch_outer:
        for _ in range(J_outer - 1):
            o_loss = outer_loss(params, hparams)
            grad_outer_w_1, grad_outer_hparams_1 = get_outer_gradients(
                o_loss, params, hparams, retain_graph=False
            )
            for g, g1 in zip(grad_outer_w, grad_outer_w_1):
                g += g1
            for g, g1 in zip(grad_outer_hparams, grad_outer_hparams_1):
                g += g1

        for g in grad_outer_w:
            g /= J_outer
        for g in grad_outer_hparams:
            g /= J_outer

    if stoch_inner:

        def w_updated():
            return w_update_f(params, hparams)

    else:
        w_new = w_update_f(params, hparams)

        def w_updated():
            return w_new

    def compute_and_set_grads(vs):
        Jfp_mapTv = torch_grad(
            w_updated(), params, grad_outputs=vs, retain_graph=not stoch_inner
        )

        for v, jtv, g in zip(vs, Jfp_mapTv, grad_outer_w):
            v.grad = torch.zeros_like(v)
            v.grad += v_update(v, jtv, g)

    if linsys_start is not None:
        vparams = [l.detach().clone() for l in linsys_start]
    else:
        vparams = [gw.detach().clone() for gw in grad_outer_w]

    if optim_build is None:
        optim = torch.optim.SGD(vparams, lr=1.0)
        scheduler = None
    else:
        if opt_params is None:
            optim, scheduler = optim_build(vparams)
        else:
            optim, scheduler = optim_build(vparams, **opt_params)

    # Solve the linear system
    for i in range(K):
        vparams_prev = [v.detach().clone() for v in vparams]
        optim.zero_grad()
        compute_and_set_grads(vparams)
        optim.step()
        if scheduler:
            scheduler.step()
        if verbose and ((K < 5) or (i % (K // 5) == 0 or i == K - 1)):
            print(
                f"k={i}: linsys, ||v - v_prev|| = {[torch.norm(v - v_prev).item() for v, v_prev in zip(vparams, vparams_prev)]}"
            )

    if any(
        [(torch.isnan(torch.norm(v)) or torch.isinf(torch.norm(v))) for v in vparams]
    ):
        raise ValueError("Hypergradient's linear system diverged!")

    grads_indirect = [torch.zeros_like(g) for g in hparams]

    # Compute Jvp w.r.t lambda
    for i in range(J_inner):
        retain_graph = (not stoch_inner) and (i < J_inner - 1)

        djac_wrt_lambda = torch_grad(
            w_updated(),
            hparams,
            grad_outputs=vparams,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        for g, g1 in zip(grads_indirect, djac_wrt_lambda):
            if g1 is not None:
                g += g1 / J_inner

    grads = [g + v for g, v in zip(grad_outer_hparams, grads_indirect)]

    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads, vparams


def grad_unused_zero(
    output, inputs, grad_outputs=None, retain_graph=False, create_graph=False
):
    grads = torch.autograd.grad(
        output,
        inputs,
        grad_outputs=grad_outputs,
        allow_unused=True,
        retain_graph=retain_graph,
        create_graph=create_graph,
    )

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))


def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=True)
    grad_outer_hparams = grad_unused_zero(
        outer_loss, hparams, retain_graph=retain_graph
    )

    return grad_outer_w, grad_outer_hparams


def update_tensor_grads(hparams, grads):
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g
