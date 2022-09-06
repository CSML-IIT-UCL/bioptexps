import os
from math import ceil

from utils import get_freer_gpu

n_gpu = get_freer_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = f"{n_gpu}"

from itertools import repeat

import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import time
import data

import torch.nn.functional as F
import scipy

from utils import (
    CustomTensorIterator,
    Logger,
    project_onto_l1_ball,
    set_seed,
    get_freer_gpu,
)
from bilevel_optimizers import TorchBiOptimizer

# hyperparameters
exp_name = ""
mnist_path = "../data"

budget_mode = "gradients"
budget = 2000000  # budget is n_steps * (T + K) * train_batch_size + (K + K_2) or seconds depending on budget_mode
dataset = "mnist"  # 'twentynews', 'aloi', 'rcv1', 'mnist'
subsample = (
    None  # number of  train + validation example, if None take the whole dataset
)
# subsample = 10000
val_size = 0.25
data_seed = (
    1  # Keep it fixed to have the same split train-val every time, otherwise put None
)
algo_seed = 1  # Keep it fixed to have the same minibatch composition every time, otherwise put None
n_train_logs = 5
n_checkpoints = 1
save_params = False
val_log_interval = 1
use_cuda = True
cuda = use_cuda and torch.cuda.is_available()

outer_opt = "SGD"  # Used to update the features of the poisoned examples
outer_lr = 4e08
outer_mu = 0.0
inner_opt_class = "SGD"  # Used to update the parameters of the classification model
inner_lr_start = 0.09  # initial inner learning rate
inner_lr_gamma = None  # if not None the inner lr at iteration t is inner_lr_start*inner_lr_gamma/(inner_lr_gamma + t)
inner_mu = 0.0  # momentum parameter for the inner optimizer
reverse_lr_start = "inner_lr_start"  # initial learning rate for the linear system (LS)
reverse_lr_gamma = "inner_lr_gamma"  # LS lr at iteration t is reverse_lr_start*reverse_lr_gamma/(reverse_lr_gamma + t)
reverse_mu = 0.0  # momentum parameter for the LS optimizer
T = "300"  # number of iterations of the inner optimizer
K = "T"  # number of iterations for the linear system
J = "K"  # minibatch size to compute the cross double derivative of the inner objective
T_inc_rate = (
    0.002  # rate of increment for T, used only if T is either 'inc_lin' or 'inc_sqrt'.
)
K_inc_rate = "T_inc_rate"  # rate of increment for K, used only if K is either 'inc_lin' or 'inc_sqrt'.
J_inc_rate = "T_inc_rate"  # rate of increment for J, used only if J is either 'inc_lin' or 'inc_sqrt'.
warm_start = False  # if True uses warm-start on the inner problem
warm_start_linsys = False  # if True uses warm-start on the linear system solved to compute the hypergradient

reg_param = 1e-1
bias = False  # if True, add the bias to the linear model used for classification

poison_size = 0.2  # poisoned examples as a fraction of the entire training set
poison_init_type = "zero"  # start from random training examples if "zero" or add to them a small noise if random
poison_constraint_type = "L2"  # can be  "Linf", "L2", "L1"
poison_max_linf_norm = 2e-1  # radius of the ball for the projection in infinity norm
poison_max_l2_norm = 5  # radius of the ball for the projection in L2 norm
poison_max_l1_norm = 50  # radius of the ball for the projection in L1 norm

train_batch_size = 90  # number of examples in minibatch for training, -1 for full batch
val_batch_size = (
    None  # number of examples in minibatch for validation, None for full batch
)
sparse_data = False  # set to True if the dataset contains sparse features
hypergrad_clip = None


# Parameters of the final training with optimal poisoned examples.

T_final = int(1e3)  # number of iteration of the inner optimizer
inner_lr_final = 0.1
inner_mu_final = 0.5
reg_param_final = 1e-1
train_val = (
    False  # if true use both the training and validation set as the final train set
)
final_batch_size = (
    None  # number of examples in minibatch for final training, None for full batch
)


# PyTorch helper functions

default_tensor_str = "torch.cuda.FloatTensor" if cuda else "torch.FloatTensor"

kwargs = {"num_workers": 0, "pin_memory": False} if cuda else {}
torch.set_default_tensor_type(default_tensor_str)


# torch.multiprocessing.set_start_method('forkserver')


def frnp(x):
    return torch.from_numpy(x).cuda().float() if cuda else torch.from_numpy(x).float()


def tonp(x, cuda=cuda):
    return x.detach().cpu().numpy() if cuda else x.detach().numpy()


# Utility functions


def from_maybe_sparse(x):
    if not scipy.sparse.issparse(x):
        return torch.FloatTensor(x)

    x = x.tocoo()
    values = x.data
    indices = np.vstack((x.row, x.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = x.shape

    s = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    # s = s.to_sparse_csr()
    return s


def project(hparams):
    for hp in hparams:
        if poison_constraint_type == "Linf":
            hp.data.clamp_(-poison_max_linf_norm, poison_max_linf_norm)
        elif poison_constraint_type == "L2":
            cond = torch.norm(hp.data, dim=1) > poison_max_l2_norm
            hp.data[cond] = (
                poison_max_l2_norm
                * hp.data[cond]
                / torch.norm(hp.data, dim=1)[cond].unsqueeze(1)
            )
        elif poison_constraint_type == "L1":
            hp.data = project_onto_l1_ball(hp, poison_max_l1_norm)

        hp.data.clamp_(-x_poison, 1 - x_poison)


def plot_accuracy(x, ys: dict, title="accuracy"):
    plt.title(title)
    for y_name, y in ys.items():
        plt.plot(x, y, label=y_name)
    plt.legend()
    plt.xlabel("budget")
    plt.savefig(f"{title}.pdf")
    plt.show()
    plt.close()


def show_poisoned_images(xp, noise, length=1):
    fig, axes = plt.subplots(length, 3)
    fig.suptitle(
        f"{length} poisoned images: original (left), noise (center), poisoned (rigth)"
    )
    img_dim = int(np.sqrt(xp.shape[1]))

    for i in range(length):
        xp_np = tonp(xp.detach())
        xp_img = xp_np.reshape(-1, img_dim, img_dim, 1)
        axes[i][0].imshow(xp_img[i], cmap=plt.get_cmap("gray"))

        noise_np = tonp(noise.detach())
        noise_img = noise_np.reshape(-1, img_dim, img_dim, 1)
        axes[i][1].imshow(noise_img[i], cmap=plt.get_cmap("gray"))
        axes[i][2].imshow(xp_img[i] + noise_img[i], cmap=plt.get_cmap("gray"))

    plt.show()
    plt.savefig("poisoned_images.png")
    plt.close()
    pass


# Hyperparameter processing

try:
    T = int(T)
except Exception as e:
    print(e)

K = T if K == "T" else int(K)
if J == "T":
    J = T
elif J == "K":
    J = K
else:
    J = int(J)
K_inc_rate = T_inc_rate if K_inc_rate == "T_inc_rate" else K_inc_rate
J_inc_rate = T_inc_rate if J_inc_rate == "T_inc_rate" else J_inc_rate
reverse_lr_start = (
    inner_lr_start if reverse_lr_start == "inner_lr_start" else reverse_lr_start
)
reverse_lr_gamma = (
    inner_lr_gamma if reverse_lr_gamma == "inner_lr_gamma" else reverse_lr_gamma
)

if T == "inc_lin" and T_inc_rate is not None:
    T = lambda o: ceil(T_inc_rate * (o + 1))
elif T == "inc_sqrt" and T_inc_rate is not None:
    T = lambda o: ceil(T_inc_rate * np.sqrt(o + 1))
elif isinstance(T, str):
    T = int(T)

if K == "inc_lin" and T_inc_rate is not None:
    K = lambda o: ceil(K_inc_rate * (o + 1))
elif K == "inc_sqrt" and T_inc_rate is not None:
    K = lambda o: ceil(K_inc_rate * np.sqrt(o + 1))
elif isinstance(K, str):
    K = int(K)

if J == "inc_lin" and T_inc_rate is not None:
    J = lambda o: ceil(J_inc_rate * (o + 1))
elif J == "inc_sqrt" and T_inc_rate is not None:
    J = lambda o: ceil(J_inc_rate * np.sqrt(o + 1))
elif isinstance(J, str):
    J = int(J)

train_batch_size = None if train_batch_size == -1 else train_batch_size
val_batch_size = None if val_batch_size == -1 else val_batch_size

print(f"J={J}, T={T}, K={K}, train_batch_size={train_batch_size}")


if inner_lr_gamma is not None:

    def inner_lr(outer_step, inner_step):
        return inner_lr_start * inner_lr_gamma / (inner_step + inner_lr_gamma)

else:
    inner_lr = inner_lr_start

if reverse_lr_gamma is not None:

    def reverse_lr(outer_step, inner_step):
        return reverse_lr_start * reverse_lr_gamma / (inner_step + reverse_lr_gamma)

else:
    reverse_lr = reverse_lr_start

train_log_interval = lambda x: x // n_train_logs if x > n_train_logs - 1 else x
save_budget_interval = budget // n_checkpoints

l = Logger(writer=SummaryWriter(), console=True)

set_seed(torch, np, seed=data_seed, is_deterministic=False)


if dataset == "twentynews":
    X, x_test, y, y_test = data.load_twentynews()
if dataset == "aloi":
    X, x_test, y, y_test = data.load_aloi()
if dataset == "rcv1":
    X, x_test, y, y_test = data.load_rcv1()
if dataset == "mnist":
    X, x_test, y, y_test = data.load_mnist(path=mnist_path)

x_train, x_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=val_size)

if subsample is not None:
    subsample_size_train = min(1, int(subsample * (1 - val_size)) / len(y_train))
    subsample_size_val = min(1, int(subsample * val_size) / len(y_val))
    x_train, _, y_train, _, = train_test_split(
        x_train, y_train, stratify=y_train, train_size=subsample_size_train
    )
    (
        x_val,
        _,
        y_val,
        _,
    ) = train_test_split(x_val, y_val, stratify=y_val, train_size=subsample_size_val)


x_poison, x_train, y_poison, y_train = train_test_split(
    x_train, y_train, stratify=y_train, train_size=poison_size
)

train_samples, n_features = [x + y for x, y in zip(x_train.shape, x_poison.shape)]
test_samples, n_features = x_test.shape
poison_samples, n_features = x_poison.shape
val_samples, n_features = x_val.shape
n_classes = np.unique(y).shape[0]

print(
    "Dataset %s, poison_samples=%i train_samples=%i, val_samples=%i, test_samples=%i, n_features=%i, n_classes=%i"
    % (
        dataset,
        poison_samples,
        train_samples,
        val_samples,
        test_samples,
        n_features,
        n_classes,
    )
)

xs = (x_train, x_val, x_test, x_poison)


if cuda:
    xs = [from_maybe_sparse(x).cuda() for x in xs]
else:
    xs = [from_maybe_sparse(x) for x in xs]

if not sparse_data and scipy.sparse.issparse(x_train):
    xs = [x.to_dense() for x in xs]

x_train, x_val, x_test, x_poison = xs
y_train, y_val, y_test, y_poison = (
    frnp(y_train).long(),
    frnp(y_val).long(),
    frnp(y_test).long(),
    frnp(y_poison).long(),
)

# Init training

set_seed(torch, np, seed=algo_seed, is_deterministic=False)

if poison_init_type == "random":
    hparams = [(1000 * torch.randn_like(x_poison)).requires_grad_(True)]
elif poison_init_type == "zero":
    hparams = [(1000 * torch.zeros_like(x_poison)).requires_grad_(True)]
else:
    raise NotImplementedError

project(hparams)

y_train_poisoned = torch.cat([y_train, y_poison], dim=0)

# torch.DataLoader is not efficient with sparse tensors on GPU
train_batch_size = (
    len(y_train_poisoned) if train_batch_size is None else int(train_batch_size)
)
val_batch_size = len(y_val) if val_batch_size is None else int(val_batch_size)

iterators = []
for bs, y in [
    (train_batch_size, y_train_poisoned),
    (train_batch_size, y_train_poisoned),
    (val_batch_size, y_val),
]:
    if bs < len(y):
        print("making iterator with batch size ", bs)
        iterators.append(
            CustomTensorIterator(
                [torch.arange(len(y))], batch_size=bs, shuffle=True, **kwargs
            )
        )
    else:
        iterators.append(repeat(torch.arange(len(y))))

train_iterator, reverse_iterator, val_iterator = iterators
reverse_iterator = train_iterator

w = torch.zeros(n_features, n_classes).requires_grad_(True)  # linear model weights

if bias:
    b = torch.zeros(n_classes).requires_grad_(True)  # linear model biases
    parameters = (w, b)
else:
    parameters = (w,)


def out_f(x, params):
    out = x @ params[0]
    out += params[1] if len(params) == 2 else 0
    return out


def val_loss(opt_params, hparams=None, data=None):
    x_mb, y_mb = x_val[data], y_val[data]
    out = out_f(x_mb, opt_params[: len(parameters)])
    val_loss = -F.cross_entropy(out, y_mb)
    # pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    # acc = pred.eq(y_mb.view_as(pred)).sum().item() / len(y_mb)

    # val_losses.append(tonp(val_loss))
    # val_accs.append(acc)
    return val_loss


def eval(params, x, y):
    with torch.no_grad():
        out = out_f(x, params)
        loss = F.cross_entropy(out, y)
        pred = out.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        acc = pred.eq(y.view_as(pred)).sum().item() / len(y)

        return tonp(loss), acc


if outer_opt == "SGD":
    outer_opt = torch.optim.SGD
    outer_opt_kwargs = dict(momentum=outer_mu)
elif outer_opt == "ADAM":
    outer_opt = torch.optim.Adam
    outer_opt_kwargs = dict()
else:
    raise NotImplementedError

if inner_opt_class == "SGD":
    inner_opt = torch.optim.SGD
    inner_opt_kwargs = dict(momentum=inner_mu)
elif inner_opt_class == "ADAM":
    inner_opt = torch.optim.Adam
    inner_opt_kwargs = dict()
else:
    raise NotImplementedError

linsys_opt = torch.optim.SGD
linsys_opt_kwargs = dict(momentum=reverse_mu)

stochastic_train = True if train_batch_size < len(y_train) else False
stochastic_val = True if val_batch_size < len(y_val) else False


def after_hypergrad_callback(hparams):
    if hypergrad_clip is not None:
        torch.nn.utils.clip_grad_norm_(hparams, hypergrad_clip)


biopt_kwargs = dict(
    inner_parameters=parameters,
    outer_parameters=hparams,
    inner_optimizer=inner_opt,
    outer_optimizer=outer_opt,
    inner_batch_size=train_batch_size,
    outer_batch_size=val_batch_size,
    T=T,
    K=K,
    J=J,
    inner_iterator=train_iterator,
    outer_iterator=val_iterator,
    reverse_iterator=reverse_iterator,
    inner_lr=inner_lr,
    outer_lr=outer_lr,
    linsys_lr=reverse_lr,
    inner_optimizer_kwargs=inner_opt_kwargs,
    outer_optimizer_kwargs=outer_opt_kwargs,
    linsys_optimizer=linsys_opt,
    linsys_optimizer_kwargs=linsys_opt_kwargs,
    warm_start_linsys=warm_start_linsys,
    stoch_outer=stochastic_val,
    stoch_inner=True,
    after_hypergrad_callback=after_hypergrad_callback,
    warm_start=warm_start,
)

bi_opt = TorchBiOptimizer(**biopt_kwargs)

val_losses, val_accs = [], []
test_losses, test_accs = [], []
train_losses, train_accs = [], []
budgets = []
check_point_hparams = []
check_point_budgets = []
prev_hparams = None
total_time = 0
current_budget = 0
current_budget_from_last_save = 0
n_saves = 0
while current_budget < budget:
    start_time = time.time()
    outer_step = bi_opt.outer_step_count

    x_train_poisoned = torch.cat([x_train, x_poison + hparams[0]], dim=0)

    def train_loss(params, hparams, data):
        x_train_poisoned = torch.cat([x_train, x_poison + hparams[0]], dim=0)
        x_mb, y_mb = x_train_poisoned[data], y_train_poisoned[data]
        out = out_f(x_mb, params)
        return F.cross_entropy(out, y_mb) + reg_param * (params[0] ** 2).mean()

    bi_opt.init_inner()
    inner_losses = []

    for t in range(bi_opt.T()):
        start = time.time()

        t_loss = bi_opt.inner_step(inner_loss=train_loss)

        if torch.isnan(t_loss):
            print("loss is NaN! ending training and restoring last val")
            hparams = prev_hparams
            break

        # params_history.append([p.clone() for p in parameters])
        inner_losses.append(t_loss.detach())

        if t % train_log_interval(bi_opt.T()) == 0 or t == bi_opt.T() - 1:
            print("t={} loss: {} ({:.4f}s)".format(t, t_loss, time.time() - start))

    if torch.isnan(t_loss):
        break

    prev_hparams = [hp.detach().clone() for hp in hparams]

    bi_opt.outer_step(outer_loss=val_loss, inner_loss=train_loss)
    project(hparams)

    iter_time = time.time() - start_time
    total_time += iter_time

    if outer_step % val_log_interval == 0 or outer_step == bi_opt.T() - 1:
        test_loss, test_acc = eval(parameters, x_test, y_test)
        v_loss, v_acc = eval(parameters, x_val, y_val)
        t_loss, t_acc = eval(parameters, x_train_poisoned, y_train_poisoned)
        val_losses.append(v_loss)
        val_accs.append(v_acc)
        train_losses.append(t_loss)
        train_accs.append(t_acc)
        budgets.append(current_budget)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        step = current_budget
        print(f"step: {step}")
        l.add_scalar("outer_step", outer_step, step)
        l.add_scalar("current_budget", current_budget, step)
        l.add_scalar("iter_time", iter_time, step)
        l.add_scalar("hgrad_nom", sum([tonp(torch.norm(h.grad)) for h in hparams]))
        l.add_scalar("train_loss", t_loss, step)
        l.add_scalar("train_acc", 100 * t_acc, step)
        l.add_scalar("val_loss", v_loss, step)
        l.add_scalar("val_acc", 100 * v_acc, step)
        l.add_scalar("test_loss", test_loss, step)
        l.add_scalar("test_acc", 100 * test_acc, step)
        l.add_scalar("total_time", total_time, step)

    if budget_mode == "gradients":
        current_budget = bi_opt.get_budget()
    if budget_mode == "time":
        current_budget += iter_time

    if current_budget >= budget or (current_budget // save_budget_interval >= n_saves):
        n_saves += 1
        print("checkpoint: saving data")
        check_point_hparams.append([hp.detach().clone() for hp in prev_hparams])
        check_point_budgets.append(current_budget)
        if save_params:
            torch.save(prev_hparams, f"hparams_{n_saves}_{current_budget}")
            torch.save(parameters, f"params_{n_saves}_{current_budget}")
        if dataset == "mnist":
            # show_poisoned_images(x_poison, prev_hparams[0], length=3)
            pass

        current_budget_from_last_save = 0


print("HPO ended in {:.2e} seconds\n".format(total_time))
if dataset == "mnist":
    show_poisoned_images(x_poison, prev_hparams[0], length=3)

plot_accuracy(
    budgets,
    dict(train_accuracy=train_accs, val_accuracy=val_accs, test_accuracy=test_accs),
    title="accuracy",
)


val_losses, val_accs = [], []
test_losses, test_accs = [], []
train_losses, train_accs = [], []
for o_iter, (hparams, budget) in enumerate(
    zip(check_point_hparams, check_point_budgets)
):
    print(f"Final training on checkpoint {o_iter}/{n_checkpoints}")
    # print(f'at budget={best_budget} and outer iter ={best_outer_iter}')

    inner_lr_f = (
        (lambda x: inner_lr_final(None, x))
        if callable(inner_lr_final)
        else lambda x: inner_lr_final
    )

    class Scheduler:
        linsys_lr = inner_lr_f
        first_lr = inner_lr_f(1)

        def __init__(self, optimizer):
            self.count = 2
            self.optimizer = optimizer

        def step(self):
            lr = Scheduler.linsys_lr(self.count)
            # print(f"new_lr {lr}")
            if isinstance(self.optimizer, torch.optim.SGD):
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
            self.count += 1
            return lr

    w = torch.zeros(n_features, n_classes).requires_grad_(True)

    if bias:
        b = torch.zeros(n_classes).requires_grad_(True)
        opt_params = (w, b)
    else:
        opt_params = (w,)

    if inner_opt_class == "SGD":
        inner_opt = torch.optim.SGD(
            opt_params, lr=inner_lr_f(1), momentum=inner_mu_final
        )
    elif inner_opt_class == "ADAM":
        inner_opt = torch.optim.Adam(opt_params, lr=inner_lr_f(1))
    else:
        NotImplementedError

    scheduler = Scheduler(inner_opt)

    x_train_poisoned = torch.cat([x_train, x_poison + hparams[0]], dim=0)
    y_train_poisoned = torch.cat([y_train, y_poison], dim=0)

    if train_val:
        x_train_val = torch.cat([x_val, x_train_poisoned], dim=0)
        y_train_val = torch.cat([y_val, y_train_poisoned], dim=0)
    else:
        x_train_val = x_train_poisoned
        y_train_val = y_train_poisoned

    final_batch_size = (
        len(y_train_val) if final_batch_size is None else final_batch_size
    )

    if final_batch_size < len(y_train_val):
        print("making iterator with batch size ", final_batch_size)
        train_val_iterator = CustomTensorIterator(
            [torch.arange(len(y_train_val))], batch_size=bs, shuffle=True, **kwargs
        )
    else:
        train_val_iterator = repeat(torch.arange(len(y_train_val)))

    def train_loss(
        params, hparams, data=None
    ):  # hyperparameters are inside data (training examples)
        x_mb, y_mb = x_train_val[data], y_train_val[data]
        out = out_f(x_mb, params)
        return F.cross_entropy(out, y_mb) + reg_param_final * (params[0] ** 2).mean()

    for t in range(T_final):
        inner_opt.zero_grad()
        train_val_minibatch = next(train_val_iterator)
        t_loss = train_loss(opt_params, hparams, data=train_val_minibatch)
        t_loss.backward()
        inner_opt.step()
        scheduler.step()

        if t % (T_final // 40) == 0 or t == T_final - 1:
            t_loss, train_acc = eval(opt_params, x_train_poisoned, y_train_poisoned)
            val_loss, val_acc = eval(opt_params, x_val, y_val)
            test_loss, test_acc = eval(opt_params, x_test, y_test)
            step = (t + 1) * final_batch_size
            print(f"step: {step}")
            l.add_scalar(f"final_train_loss_{o_iter}", t_loss, step)
            l.add_scalar(f"final_train_acc_{o_iter}", 100 * train_acc, step)
            l.add_scalar(f"final_val_loss_{o_iter}", val_loss, step)
            l.add_scalar(f"final_val_acc_{o_iter}", 100 * val_acc, step)
            l.add_scalar(f"final_test_loss_{o_iter}", test_loss, step)
            l.add_scalar(f"final_test_acc_{o_iter}", 100 * test_acc, step)

            if t == T_final - 1:
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                val_accs.append(val_acc)
                l.add_scalar("final_train_loss", t_loss, budget)
                l.add_scalar("final_train_acc", 100 * train_acc, budget)
                l.add_scalar("final_val_loss", val_loss, budget)
                l.add_scalar("final_val_acc", 100 * val_acc, budget)
                l.add_scalar("final_test_loss", test_loss, budget)
                l.add_scalar("final_test_acc", 100 * test_acc, budget)


l.add_scalar("loss", val_acc, budget)  # loss to minimize for guild HPO

plot_accuracy(
    check_point_budgets,
    dict(train_accuracy=train_accs, val_accuracy=val_accs, test_accuracy=test_accs),
    title="final_accuracy",
)
