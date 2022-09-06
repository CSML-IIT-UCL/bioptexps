import random
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import os
from utils import set_seed, bettercycle, get_freer_gpu
import itertools as it

n_gpu = get_freer_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = f"{n_gpu}"


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from hypergrad import stoch_AID

import pandas as pd

EXP_DIR = Path("../exps/deq/no_momentum_whole_train_proj05_T3/")


TARGET_DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def main():
    run_comparison()
    # run()
    # plot_results()


def run_comparison(
    base_budget=10000000,
    # base_budget = 10000,
    hparams_dict=None,
):
    if hparams_dict is None:
        hparams_dict = dict(
            # dummy=(True,),
            seed=list(range(5)),
            batch_size=(600, 6000, 60000),
            # batch_size=(500, 50),
            warm_start=(True, False),
            # lr=(.1, .7, .5, 0.3),
            lr=(0.5,),
            model=("ff",),
        )

    res_df_all = None

    budget_dict = {
        60: base_budget // 8,
        600: base_budget,
        6000: 8 * base_budget,
        60000: 20 * base_budget,
    }

    all_hparams_names = sorted(hparams_dict)
    for hparams in it.product(*(hparams_dict[name] for name in all_hparams_names)):
        hparams_dict_config = {k: v for k, v in zip(all_hparams_names, hparams)}
        hparams_label = {
            k: hparams_dict_config[k]
            for k in [
                k
                for k in hparams_dict_config
                if k != "seed" and len(hparams_dict[k]) > 1
            ]
        }

        print("evaluating config:\n", hparams_dict_config)

        batch_size = hparams_dict_config["batch_size"]
        budget = budget_dict[batch_size] if batch_size in budget_dict else base_budget
        log_interval_budget = budget // 100
        n_outer = budget // batch_size

        log_interval = np.ceil(log_interval_budget / batch_size)
        res_df, parameters = run(
            n_outer_iter=n_outer,
            log_interval=log_interval,
            **hparams_dict_config,
        )

        params_np = to_numpy(parameters)
        EXP_DIR.mkdir(parents=True, exist_ok=True)
        np.save(
            EXP_DIR / f"{hparams_dict_config}_params.npy", params_np, allow_pickle=True
        )

        res_df["label"] = f"{hparams_label}"
        if res_df_all is None:
            res_df_all = res_df
        else:
            res_df_all = pd.concat([res_df_all, res_df])

    res_df_all["total_budget"] = base_budget
    res_df_all.to_csv(EXP_DIR / "deq_comp.csv")
    plot_results(dir=EXP_DIR)


def run(
    dummy=False,
    seed=1,
    i_sig=0.01,  # initialization noise magnitude
    dw=200,  # dimensionality of the hidden state (200 for fully connected, 100 for the convolution)
    lr=0.5,  # (outer) learning rate
    momentum=0.0,
    n_outer_iter=6000,
    n_train=60000,  # number of training examples
    n_valid=0,  # number of validation examples
    n_max_test=10000,
    batch_size=600,
    batch_size_eval=10000,
    # batch_size_eval = 2000,
    do_projection=True,
    # proj_radius = 0.99,
    proj_radius=0.5,  # this is also the contraction constant of the model
    warm_start=False,
    T=3,  # number of fixed points iterations (forward);
    K=3,  # number of fixed points iterations (backward);
    T_test=20,
    log_interval=50,
    double_prec=False,
    model="ff",
):
    if dummy:
        n_train = 10
        n_valid = 10
        batch_size = 1
        batch_size_eval = 1
        n_test = 10

    fparams = deepcopy(locals())
    print(fparams)

    data = load_mnist(
        seed, num_train=n_train, num_valid=n_valid, double_prec=double_prec
    )
    num_exp, dim_x = data.train.data.shape

    set_seed(torch, np, seed, is_deterministic=False)
    dtype = torch.float64 if double_prec else torch.float32

    if model == "conv":
        initial_states = TVT(
            [
                torch.zeros(d.data.shape[0], dw, 14, 14, device=TARGET_DEVICE)
                for d in data
            ]
        )
        input_model = nn.Sequential(
            View(shape=(1, 28, 28)),
            nn.Conv2d(1, dw, 3, stride=1, padding=1, bias=False).to(TARGET_DEVICE),
            nn.MaxPool2d(2, 2),
        )
        state_model = nn.Sequential(
            nn.Conv2d(dw, dw, 3, stride=1, padding=1, bias=True).to(TARGET_DEVICE),
        )
        parameters = [
            *list(state_model.parameters()),
            *list(input_model.parameters()),
            i_sig * torch.randn(dw * 7 * 7, 10, device=TARGET_DEVICE),
            torch.zeros(10, device=TARGET_DEVICE),
        ]

        pre_linear_transform = nn.Sequential(
            nn.MaxPool2d(2, 2),
            View(shape=(7 * 7 * dw,)),
        )

    elif model == "ff":
        initial_states = TVT(
            [
                torch.zeros(d.data.shape[0], dw, device=TARGET_DEVICE, dtype=dtype)
                for d in data
            ]
        )
        parameters = [
            i_sig * torch.randn(dw, dw, device=TARGET_DEVICE, dtype=dtype),  # A
            i_sig * torch.randn(dim_x, dw, device=TARGET_DEVICE, dtype=dtype),  # B
            i_sig * torch.randn(dw, device=TARGET_DEVICE, dtype=dtype),  # c
            i_sig * torch.randn(dw, 10, device=TARGET_DEVICE, dtype=dtype),
            torch.zeros(10, device=TARGET_DEVICE, dtype=dtype),
        ]

        def input_model(x):
            return x @ parameters[1] + parameters[2]

        def state_model(state):
            return state @ parameters[0]

        def pre_linear_transform(x):
            return x

    else:
        raise ValueError(f"model {model} not available!")

    set_requires_grad(parameters)

    criterion = nn.CrossEntropyLoss()

    def proj(parameters):
        A, D, b = parameters[0], parameters[-2], parameters[-1]
        other_params = [p.detach().clone() for p in parameters[1:-2]]
        try:  # perform projection
            if model == "ff":
                A_proj, svl, svl_proj = matrix_projection_on_spectral_ball(
                    A, radius=proj_radius, dtype=dtype
                )
            elif model == "conv":
                A_proj, svl, svl_proj = conv_project(A, (7, 7), radius=proj_radius)
            else:
                raise ValueError(f"model {model} not supported.")

            D_proj = torch.clamp(
                D, min=-1, max=1
            )  # this matrix should not explode for the theory to hold

            return [A_proj, *other_params, D_proj, b], svl, svl_proj

        except (ValueError, np.linalg.LinAlgError) as e:
            print("there were nans most probably")
            raise ValueError

    def get_fs(x, targets):
        def phi(state_list, params):
            state = state_list[0]
            # return [torch.tanh(state @ A + x @ B + c)]
            return [torch.tanh(input_model(x) + state_model(state))]

        def forward(initial_state, n_iter_inner, verbose=False):
            state = [initial_state]
            opt_metrics = []
            with torch.no_grad():
                for i in range(n_iter_inner):
                    prev_state = [s.detach().clone() for s in state]
                    state = phi(state, parameters)
                    optim_metric = torch.norm(state[0] - prev_state[0]) / torch.norm(
                        state[0]
                    )
                    opt_metrics.append(optim_metric.cpu().numpy())
                    if verbose:
                        print(
                            f"||w_{i+1} - w_{i}||/||w_{i+1}||, w_shape"
                            f" = {optim_metric},"
                            f" {state[0].shape}"
                        )

                return state, opt_metrics

        def linear(state_list, params):
            s_transf = pre_linear_transform(state_list[0])
            return s_transf @ params[-2] + params[-1]

        def loss(state_list, params):
            # cross entropy loss (the outer loss of the bi-level problem)
            outputs = linear(state_list, params)
            return torch.mean(criterion(outputs, targets))

        return phi, forward, linear, loss

    def evaluate(
        initial_states,
        xs,
        targets,
        n_iter_inner,
        compute_hg=False,
        batch_size=batch_size_eval,
    ):
        n_examples = min(len(targets), n_max_test)
        xs, targets = xs[:n_examples], targets[:n_examples]
        indices = list(range(len(targets)))
        tot_acc, tot_loss, tot_hgs = 0, 0, [torch.zeros_like(p) for p in parameters]
        while indices:
            idx_last = min(batch_size, len(indices))
            idxs = indices[:idx_last]
            indices = indices[idx_last:]

            x, target = (
                xs[idxs],
                targets[idxs],
            )
            phi, forward, linear, loss = get_fs(x, target)

            state, opt_metrics = forward(
                initial_states[idxs], n_iter_inner, verbose=False
            )
            outputs = linear(state, parameters)

            tot_loss += (
                to_numpy(torch.sum(criterion(outputs, target)))
                * batch_size
                / n_examples
            )
            tot_acc += (
                100
                * to_numpy(outputs.argmax(dim=1).eq(target).float().sum())
                / n_examples
            )

            if compute_hg:
                hgs, _ = stoch_AID(
                    state,
                    parameters,
                    outer_loss=loss,
                    K=n_iter_inner,
                    J_inner=1,
                    J_outer=1,
                    fp_map=phi,
                    stoch_outer=False,
                    stoch_inner=False,
                    optim_build=inner_optim_build,
                    set_grad=False,
                    verbose=False,
                )

                for h, g in zip(tot_hgs, hgs):
                    h.data = h.data + g.data * batch_size / n_examples

        if compute_hg:

            states_updated, svl, svl_proj = proj(
                [p - g for p, g in zip(parameters, tot_hgs)]
            )
            prox_map = [s - su for s, su in zip(parameters, states_updated)]

            hg_norm = sum([(hg**2).sum() for hg in to_numpy(tot_hgs)])
            prox_map_norm = sum([(pm**2).sum() for pm in to_numpy(prox_map)])

            return tot_acc, tot_loss, hg_norm, prox_map_norm, svl_proj, opt_metrics

        else:
            return tot_acc, tot_loss

    # optimizer
    opt = torch.optim.SGD(parameters, lr, momentum=momentum)

    train_dataset = torch.utils.data.TensorDataset(
        torch.arange(0, data.train[0].shape[0]),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    def inner_optim_build(params):
        return torch.optim.SGD(params, 1), None

    rand_indices = [
        np.random.choice(n_train, n_train, replace=False)
        for _ in range(n_outer_iter + 10)
    ]
    rand_indices = np.concatenate(rand_indices)

    # Training
    # for t, idx in enumerate(bettercycle(train_loader)):
    results = defaultdict(list)
    total_time = 0
    for t in range(n_outer_iter):
        start_iter_time = time.time()
        idx = rand_indices[:batch_size]
        rand_indices = rand_indices[batch_size:]
        x, targets = data.train[0][idx], data.train[1][idx]

        phi, forward, linear, loss = get_fs(x, targets)

        opt.zero_grad()

        state, opt_metric = forward(initial_states.train[idx], n_iter_inner=T)

        # compute the hypergradient (with different methods)
        # hypergradient computation
        stoch_AID(
            state,
            parameters,
            outer_loss=loss,
            K=K,
            J_inner=1,
            J_outer=1,
            fp_map=phi,
            linsys_start=None,
            stoch_outer=False,
            stoch_inner=False,
            optim_build=inner_optim_build,
            set_grad=True,
            verbose=False,
        )

        # hg.fixed_point(states[-1], parameters, K, fully_connected_fp, loss) # slightly different results, why?

        opt.step()
        if do_projection:
            proj_params, svl, svl_proj = proj(parameters)
            for p, pp in zip(parameters, proj_params):
                p.data = pp

        if warm_start:
            initial_states.train.data[idx] = state[0].detach().clone()

        total_time += time.time() - start_iter_time

        if t % log_interval == 0 or t == n_outer_iter - 1:
            # print(f"iter={t}")

            (
                train_acc,
                train_loss,
                hg_norm,
                prox_map_norm,
                svl_proj,
                opt_metrics,
            ) = evaluate(
                initial_states.train, *data.train, n_iter_inner=T_test, compute_hg=True
            )
            val_acc, val_loss = evaluate(
                initial_states.val, *data.val, n_iter_inner=T_test
            )

            opt_metric = opt_metrics[-1] / (1 - proj_radius)

            # hgs = to_numpy([l.grad for l in parameters])
            # hg_norm = sum([(hg**2).sum() for hg in hgs])

            results["iter"].append(t)
            results["time"].append(total_time)
            results["budget"].append(t * batch_size)
            results["val_acc"].append(val_acc)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["train_loss"].append(train_loss)
            results["hg_norm"].append(hg_norm)
            results["prox_map_norm"].append(prox_map_norm)
            results["opt_metric"].append(opt_metric)

            spaces = " " * (5 - len(str(t)))
            print_str = (
                f"iter={t}{spaces}: train_loss,  prox_map_norm, hg_norm, opt_metric,  Train, Val, Test Acc, "
                f"{train_loss:.2e},  {prox_map_norm:.2e}, {hg_norm:.2e}, {opt_metric:.2e},"
                f"  {train_acc:.2f}, {val_acc:.2f}"
            )
            if not dummy:
                test_acc, test_loss = evaluate(
                    initial_states.test, *data.test, n_iter_inner=T_test
                )
                results["test_acc"].append(test_acc)
                results["test_loss"].append(test_loss)
                print_str += f", {test_acc:.2f}"

            print(print_str)

    results = pd.DataFrame(results)
    for k, v in fparams.items():
        results[k] = v
    return results, parameters


# --------------------------------------------
# UTILS
# --------------------------------------------


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f"View{self.shape}"

    def forward(self, input):
        """
        Reshapes the input according to the shape saved in the view data structure.
        """
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out


def plot_results(
    metrics=(
        "prox_map_norm",
        "hg_norm",
        "train_acc",
        "test_acc",
        "train_loss",
        "test_loss",
    ),
    x_names=("iter", "time", "budget"),
    dir=EXP_DIR,
):
    res_df = pd.read_csv(dir / "deq_comp.csv")
    labels = res_df["label"].drop_duplicates().values
    for m in metrics:
        for x_name in x_names:
            for label in labels:
                r = res_df.loc[res_df["label"] == label].groupby(by="iter")
                x = r[x_name].mean().values

                plt.plot(x, r[m].mean().values, label=label)
                plt.fill_between(
                    x=x, y1=r[m].max().values, y2=r[m].min().values, alpha=0.2
                )
            plt.title(f"{m}")
            plt.yscale("log")
            if "acc" in m:
                plt.yscale("linear")
                if x_name == "time":
                    plt.xlim([0, 250])
                elif x_name == "iter":
                    plt.xlim([0, 5000])
                elif x_name == "budget":
                    plt.xlim([0, 8e7])

            if "train_acc" == m:
                plt.ylim([92, 100.05])

            elif "acc" in m:
                plt.ylim([95, 98.1])

            plt.xlabel(x_name)
            plt.legend()

            if dir is not None:
                plot_name = f"{m}_{x_name}.png"
                plt.savefig(dir / plot_name)
            plt.show()


def to_numpy(tensor):
    if isinstance(tensor, (list, tuple)):
        return [to_numpy(v) for v in tensor]
    else:
        return tensor.detach().to(torch.device("cpu")).numpy()


def set_requires_grad(lst):
    [l.requires_grad_(True) for l in lst]


def accuracy(preds, targets):
    """Computes the accuracy"""
    return preds.argmax(dim=1).eq(targets).float().mean()


class NamedLists(list):
    def __init__(self, lst, names) -> None:
        super().__init__(lst)
        assert len(lst) == len(names)
        self.names = names

    def __getitem__(self, i):
        if isinstance(i, str):
            return self.__getattribute__(i)
        else:
            return super().__getitem__(i)


class TVT(NamedLists):  # train val & test
    def __init__(self, lst) -> None:
        super().__init__(lst, ["train", "val", "test"])
        self.train, self.val, self.test = lst


class DT(NamedLists):  # data & targets
    def __init__(self, lst) -> None:
        super().__init__(lst, ["data", "targets"])
        self.data, self.targets = lst


def load_mnist(seed=0, num_train=50000, num_valid=10000, double_prec=False):
    """Load MNIST dataset with given number of training and validation examples"""
    from torchvision import datasets

    rnd = np.random.RandomState(seed)
    mnist_train = datasets.MNIST("../data", download=True, train=True)
    train_indices = rnd.permutation(list(range(60000)))
    dta, targets = mnist_train.data, mnist_train.targets

    # print(train_indices)
    tr_inds = train_indices[:num_train]
    mnist_tr1 = DT([dta[tr_inds], targets[tr_inds]])

    val_inds = train_indices[num_train : num_train + num_valid]
    mnist_valid = DT([dta[val_inds], targets[val_inds]])

    mnist_test = datasets.MNIST("../data", download=True, train=False)

    dtype = np.float32 if not double_prec else np.float64

    def _process_dataset(dts):
        dt, tgt = np.array(dts.data.numpy(), dtype=dtype), dts.targets.numpy()
        return DT(
            [
                torch.from_numpy(np.reshape(dt / 255.0, (-1, 28 * 28))).to(
                    TARGET_DEVICE
                ),
                torch.from_numpy(tgt).to(TARGET_DEVICE),
            ]
        )

    return TVT([_process_dataset(dtt) for dtt in [mnist_tr1, mnist_valid, mnist_test]])


def matrix_projection_on_spectral_ball(a, radius=0.99, dtype=torch.FloatTensor):
    A = a.detach()
    if A.is_cuda:
        A = A.cpu()
    A = A.numpy()
    U, S, V = np.linalg.svd(A)
    S1 = np.minimum(S, radius)
    a = U @ np.diag(S1) @ V

    return torch.from_numpy(a).type(dtype).to(TARGET_DEVICE).requires_grad_(True), S, S1


# beautiful kernel conversions functions form torch to tensorflow... yeah!


def kernel_th_to_tf(kernel):
    return np.transpose(kernel, (2, 3, 1, 0))


def kernel_tf_to_th(kernel):
    return np.transpose(kernel, (3, 2, 0, 1))


def conv_project(filtr, inp_shape, radius=0.999):
    # adapted code from https://github.com/brain-research/conv-sv
    fltr = filtr.detach()
    if fltr.is_cuda:
        fltr = fltr.cpu()
    fltr = kernel_th_to_tf(fltr.numpy())

    # compute the singular values using FFT
    # first compute the transforms for each pair of input and output channels
    transform_coeff = np.fft.fft2(fltr, inp_shape, axes=[0, 1])

    # now, for each transform coefficient, compute the singular values of the
    # matrix obtained by selecting that coefficient for
    # input-channel/output-channel pairs
    U, D, V = np.linalg.svd(transform_coeff, compute_uv=True, full_matrices=False)
    # print('doing_progection')
    D_clipped = np.minimum(D, radius)
    # print(np.max(D_clipped))

    if fltr.shape[2] > fltr.shape[3]:
        clipped_transform_coeff = np.matmul(U, D_clipped[..., None] * V)
    else:
        clipped_transform_coeff = np.matmul(U * D_clipped[..., None, :], V)
    clipped_filter = np.fft.ifft2(clipped_transform_coeff, axes=[0, 1]).real
    args = [range(d) for d in fltr.shape]
    cf = clipped_filter[np.ix_(*args)]
    cf = kernel_tf_to_th(cf)
    # print(time.time() - st)  # projection is quite fast..  < 0.005 s
    # noinspection PyTypeChecker
    return (
        torch.from_numpy(cf)
        .type(torch.FloatTensor)
        .to(TARGET_DEVICE)
        .requires_grad_(True),
        np.flip(np.sort(D.flatten()), 0),
        np.flip(np.sort(D_clipped.flatten()), 0),
    )


if __name__ == "__main__":
    main()
