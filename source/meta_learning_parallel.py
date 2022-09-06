import copy
import os

from torchmeta.toy import Sinusoid

from utils import get_freer_gpu, bettercycle

n_gpu = get_freer_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = f"{n_gpu}"

import math
import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmeta.datasets import MiniImagenet, Omniglot

from torchmeta.transforms import Categorical, Rotation, ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader

# import higher
from torchvision.transforms import ToTensor, Resize, Compose

from hypergrad import stoch_AID
from utils import Logger


def main():

    parser = argparse.ArgumentParser(description="Meta-learning")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-budget", type=int, default=50000)
    parser.add_argument("--n-train-tasks", type=int, default=None)
    parser.add_argument("--n-test-tasks", type=int, default=1000)
    parser.add_argument("--n-ways", type=int, default=5)
    parser.add_argument(
        "--n-shots",
        type=int,
        default=5,
        help="number of shots (1 or 5 are used in benchmarks)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="../data", help="dataset directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="miniimagenet",
        help="omniglot, sinusoid, miniimagenet",
    )
    parser.add_argument("--net", type=str, default="normal", help="l2l or normal")
    parser.add_argument("--pretrained", type=str, default="False")
    parser.add_argument("--transductive", action="store_true", default=False)

    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--inner-log-interval", type=int, default=None)
    parser.add_argument("--inner-log-interval-test", type=int, default=None)

    parser.add_argument("--reg-param", type=float, default=0.01)
    parser.add_argument("--meta-batch-size", type=int, default=2)
    parser.add_argument("--meta-batch-size-test", type=int, default=2)
    parser.add_argument("--n-parallel-tasks", type=int, default=32)
    parser.add_argument("--n-outer-iter", type=int, default=None)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--T-test", type=int, default=10)
    # parser.add_argument('--outer-optim', type=str, default="SGD")
    # parser.add_argument('--outer-lr', type=float, default=.2)
    # parser.add_argument('--outer-optim', type=str, default="ADAM")
    parser.add_argument("--outer-optim", type=str, default="SGD")
    parser.add_argument("--outer-lr", type=float, default=0.2)
    parser.add_argument("--inner-lr", type=float, default=0.05)
    parser.add_argument(
        "--warm-start", type=str, default="No", metavar="N", help="No, Naive, Full"
    )

    args = parser.parse_args()

    n_train_tasks = args.n_train_tasks
    T, K = args.T, args.K
    reg_param = args.reg_param
    transductive = args.transductive

    inner_log_interval = args.inner_log_interval
    n_test_tasks = args.n_test_tasks  # usually 1000 tasks are used for testing
    net = args.net

    T_test = args.T_test
    outer_optim = args.outer_optim
    inner_lr = args.inner_lr
    outer_lr = args.outer_lr

    dataset_dir = args.data_dir
    dataset_name = args.dataset
    n_ways = args.n_ways
    n_shots = args.n_shots
    meta_batch_size = args.meta_batch_size
    meta_batch_size_test = args.meta_batch_size_test

    n_outer_iter = args.n_outer_iter if args.n_outer_iter is not None else 10e24
    max_budget = args.max_budget if args.max_budget is not None else 10e24
    log_interval = args.log_interval
    eval_interval = args.eval_interval

    warm_start = args.warm_start
    ws_buffer_size = n_train_tasks if n_train_tasks is not None else 10**6

    n_parallel_tasks = args.n_parallel_tasks

    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {"num_workers": 8, "pin_memory": True} if cuda else {}

    logger = Logger(writer=SummaryWriter(), console=False)

    print(
        f"Starting {n_ways}way-{n_shots}shots on {dataset_name} with "
        f"seed={args.seed}, T={T}, K={K}, mbs={meta_batch_size}, outer_lr={outer_lr}, inner_lr={inner_lr},"
        f"\n    T_test={T_test}, warm_start={warm_start}, n_train_tasks={n_train_tasks},"
        f" optimizer={outer_optim}"
    )

    # the is_deterministic flag is for reproducibility on GPUs, see https://pytorch.org/docs/master/notes/randomness.html
    set_seed(args.seed, is_deterministic=False)

    if dataset_name == "omniglot":
        omniglot_kwargs = dict(
            root=dataset_dir,
            # Number of ways
            num_classes_per_task=n_ways,
            # Resize the images to 28x28 and converts them to PyTorch tensors (from Torchvision)
            transform=Compose([Resize(84), ToTensor()]),
            # Transform the labels to integers (e.g. ("Glagolitic/character01", "Sanskrit/character14", ...) to (0, 1, ...))
            target_transform=Categorical(num_classes=n_ways),
            # Creates new virtual classes with rotated versions of the images (from Santoro et al., 2016)
            class_augmentations=Rotation([90, 180, 270]),
            dataset_transform=ClassSplitter(
                num_train_per_class=n_shots, num_test_per_class=n_shots
            ),
            download=True,
        )

        dataset = Omniglot(meta_train=True, **omniglot_kwargs)
        # test_dataset = omniglot("data", ways=n_ways, shots=n_shots, test_shots=15, meta_test=True, download=True)
        test_dataset = Omniglot(meta_test=True, **omniglot_kwargs)

        val_dataset = Omniglot(meta_val=True, **omniglot_kwargs)

        meta_model = get_cnn_omniglot(64, n_ways).to(device)
        n_hidden = 800
        criterion = F.cross_entropy

        def post_transform(x):
            return x

    elif dataset_name == "miniimagenet":
        # dataset = miniimagenet("data", ways=n_ways, shots=n_shots, test_shots=None, meta_train=True, download=True)
        # test_dataset = miniimagenet("data", ways=n_ways, shots=n_shots, test_shots=None, meta_test=True, download=True)
        miniimagenet_kwargs = dict(
            root=dataset_dir,
            # Number of ways
            num_classes_per_task=n_ways,
            # Resize the images to 28x28 and converts them to PyTorch tensors (from Torchvision)
            transform=Compose([Resize(84), ToTensor()]),
            # Transform the labels to integers (e.g. ("Glagolitic/character01", "Sanskrit/character14", ...) to (0, 1, ...))
            target_transform=Categorical(num_classes=n_ways),
            # Creates new virtual classes with rotated versions of the images (from Santoro et al., 2016)
            dataset_transform=ClassSplitter(
                num_train_per_class=n_shots, num_test_per_class=n_shots
            ),
            download=True,
        )

        dataset = MiniImagenet(meta_train=True, **miniimagenet_kwargs)

        test_dataset = MiniImagenet(meta_test=True, **miniimagenet_kwargs)

        val_dataset = MiniImagenet(meta_val=True, **miniimagenet_kwargs)

        if net == "normal":
            meta_model = get_cnn_miniimagenet(32, transductive=transductive)
            n_hidden = 800
        else:
            raise NotImplementedError
        meta_model = meta_model.to(device)
        # print(list(meta_model.parameters()))
        criterion = F.cross_entropy

        def post_transform(x):
            return x

    elif dataset_name == "sinusoid":
        n_hidden = 40
        sinusoid_kwargs = dict(
            noise_std=0,
            num_samples_per_task=n_shots + n_shots,
            dataset_transform=ClassSplitter(
                num_train_per_class=n_shots, num_test_per_class=n_shots
            ),
        )

        dataset = Sinusoid(
            num_tasks=int(n_train_tasks) if n_train_tasks is not None else 10**6,
            **sinusoid_kwargs,
        )

        test_dataset = Sinusoid(num_tasks=n_test_tasks, **sinusoid_kwargs)

        val_dataset = Sinusoid(num_tasks=n_test_tasks, **sinusoid_kwargs)
        meta_model = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        ).to(device)
        criterion = F.mse_loss
        n_ways = 1

        def post_transform(x):
            return [
                y.float() / 5.0 for y in x
            ]  # divide by 5 to put the input in the interval [-1,1]

    else:
        raise NotImplementedError(
            "DATASET NOT IMPLEMENTED! only omniglot, miniimagenet and sinusoid"
        )

    dataloader = BatchMetaDataLoader(dataset, batch_size=meta_batch_size, **kwargs)
    test_dataloader = BatchMetaDataLoader(
        test_dataset, batch_size=meta_batch_size_test, **kwargs
    )
    val_dataloader = BatchMetaDataLoader(
        val_dataset, batch_size=meta_batch_size_test, **kwargs
    )

    # Limit train tasks
    if n_train_tasks is not None:
        task_iter = iter(dataloader)
        task_list = [
            next(task_iter) for _ in range(math.ceil(n_train_tasks / meta_batch_size))
        ]
        new_dataset = {}
        new_dataset["train"] = [
            torch.cat([t["train"][0] for t in task_list], dim=0),
            torch.cat([t["train"][1] for t in task_list], dim=0),
        ]
        new_dataset["test"] = [
            torch.cat([t["test"][0] for t in task_list], dim=0),
            torch.cat([t["test"][1] for t in task_list], dim=0),
        ]

        new_dataset["train"] = [r[:n_train_tasks] for r in new_dataset["train"]]
        new_dataset["test"] = [r[:n_train_tasks] for r in new_dataset["test"]]

        class MyDataset(Dataset):
            def __init__(self):
                self.data = new_dataset

            def __getitem__(self, index):
                train = [self.data["train"][0][index], self.data["train"][1][index]]
                test = [self.data["test"][0][index], self.data["test"][1][index]]

                return {"train": train, "test": test, "index": index}

            def __len__(self):
                return self.data["train"][0].size(dim=0)

        dataloader = DataLoader(
            MyDataset(), batch_size=meta_batch_size, shuffle=True, **kwargs
        )

    eval_dict = dict(
        train=dataloader,
        val=val_dataloader,
        test=test_dataloader,
    )

    if outer_optim == "ADAM":
        outer_opt = torch.optim.Adam(params=meta_model.parameters(), lr=outer_lr)
    elif outer_optim == "SGD":
        outer_opt = torch.optim.SGD(lr=outer_lr, params=meta_model.parameters())
    else:
        raise NotImplementedError

    def inner_optim_build(params, lr=inner_lr):
        return torch.optim.SGD(params, lr=lr), None

    def get_params_init(bs=meta_batch_size):
        return [
            torch.zeros(bs, n_hidden, n_ways, requires_grad=True, device=device),
            torch.zeros(bs, n_ways, requires_grad=True, device=device),
        ]

    if warm_start == "Naive":
        ws_params = get_params_init(bs=meta_batch_size)
    elif warm_start == "Full":
        ws_params = get_params_init(bs=ws_buffer_size)
    else:
        ws_params = None

    print("Start Meta-training")
    total_time = 0
    for k, batch in enumerate(bettercycle(dataloader)):
        is_final_iter = k == n_outer_iter - 1 or k * meta_batch_size > max_budget
        start_time_step = time.time()
        meta_model.train()

        # tr_xs, tr_ys = batch["train"][0].to(device), batch["train"][1].to(device)
        # tst_xs, tst_ys = batch["test"][0].to(device), batch["test"][1].to(device)
        tr_xs, tr_ys = batch["train"][0].to(device), batch["train"][1].to(device)
        tst_xs, tst_ys = batch["test"][0].to(device), batch["test"][1].to(device)
        tr_xs, tr_ys, tst_xs, tst_ys = post_transform([tr_xs, tr_ys, tst_xs, tst_ys])

        if "index" in batch and warm_start == "Full":
            task_indexes = batch["index"].to(device)
        else:
            task_indexes = torch.arange(0, meta_batch_size)

        outer_opt.zero_grad()

        val_loss, val_acc = 0, 0
        init_time, forward_time, backward_time = 0, 0, 0

        if n_parallel_tasks is not None or n_parallel_tasks == -1:
            task_batches = []
            for i in range(int(np.ceil(tr_xs.shape[0] / n_parallel_tasks))):
                end_task_idx = min(n_parallel_tasks * (i + 1), tr_xs.shape[0])
                task_batches.append(
                    [
                        t[i * n_parallel_tasks : end_task_idx]
                        for t in (tr_xs, tr_ys, tst_xs, tst_ys, task_indexes)
                    ]
                )

        else:
            task_batches = [(tr_xs, tr_ys, tst_xs, tst_ys, task_indexes)]

        n_tasks = 0
        for (tr_xs, tr_ys, tst_xs, tst_ys, task_indexes) in task_batches:
            start_time_task = time.time()
            # batch of tasks set up
            if warm_start != "No" and task_indexes.max() < ws_buffer_size:
                params = ws_params
            else:
                params = get_params_init(bs=tr_xs.shape[0])
                task_indexes = None

            task = TaskBatch(
                reg_param,
                meta_model,
                (tr_xs, tr_ys, tst_xs, tst_ys),
                task_indexes=task_indexes,
                meta_batch_size=meta_batch_size,
                criterion=criterion,
            )

            init_time_task = time.time() - start_time_task

            inner_loop(
                params,
                inner_optim_build,
                task,
                T,
                log_interval=inner_log_interval,
            )

            forward_time_task = time.time() - start_time_task - init_time_task

            # hypergradient computation
            stoch_AID(
                params,
                list(meta_model.parameters()),
                task.val_loss_f,
                K,
                J_inner=1,
                J_outer=1,
                inner_loss=task.train_loss_f,
                linsys_start=None,
                stoch_outer=False,
                stoch_inner=False,
                optim_build=inner_optim_build,
                set_grad=True,
                verbose=False,
            )

            backward_time_task = (
                time.time() - start_time_task - init_time_task - forward_time_task
            )

            init_time += init_time_task
            forward_time += forward_time_task
            backward_time += backward_time_task

            n_tasks += task.val_losses.shape[0]
            val_loss += task.val_losses.sum()
            val_acc += (100 * task.val_accs).sum()

        val_loss /= n_tasks
        val_acc /= n_tasks

        outer_opt.step()
        step_time = time.time() - start_time_step
        total_time += step_time

        if k % log_interval == 0 or k == is_final_iter:
            budget = k * meta_batch_size
            epoch = budget // n_train_tasks if n_train_tasks is not None else 0
            print(f"step: {k}")

            print(
                f"  Val Acc, Loss: {val_acc:.2f}, {val_loss:.2e} "
                f"({total_time:.2f}s IT: {step_time:.2f}s F: {forward_time:.2f}s, B: {backward_time:.2f}s)"
            )
            logger.add_scalar("epoch", epoch, k)
            logger.add_scalar("budget", budget, k)
            logger.add_scalar("time", total_time, k)
            logger.add_scalar("val_loss", val_loss, k)
            logger.add_scalar("val_acc", val_acc, k)
            logger.add_scalar("step_time", step_time, k)
            logger.add_scalar("forward_time", forward_time, k)
            logger.add_scalar("backward_time", backward_time, k)

        if k % eval_interval == 0 or is_final_iter:

            if warm_start != "No" and k == n_outer_iter - 1:
                T_test_list = [T_test, 2 * T_test, 5 * T_test, 10 * T_test]
            else:
                T_test_list = [T_test]

            for T_t in T_test_list:
                t_str = "" if T_t == T_test else f"_{T_t}"
                print(f"  Eval with T_test={T_t}:")

                for d_name, loader in eval_dict.items():
                    meta_model_test = copy.deepcopy(meta_model)

                    start_time_eval = time.time()
                    losses, accs = evaluate(
                        n_test_tasks,
                        loader,
                        meta_model_test,
                        get_params_init,
                        T_t,
                        reg_param,
                        inner_optim_build,
                        n_parallel_tasks=n_parallel_tasks,
                        post_transform=post_transform,
                        criterion=criterion,
                    )

                    time_eval = time.time() - start_time_eval

                    loss_mean, loss_std = losses.mean(), losses.std()
                    acc_mean, acc_std = 100 * accs.mean(), (100 * accs).std()

                    spaces = " " * (9 - len(d_name))
                    print(
                        f"{spaces}{d_name} acc, loss: {acc_mean:.2f} +- {acc_std:.2f}, "
                        f"{loss_mean:.2e} +- {loss_std:.2e} "
                        f"({time_eval:.2f}s, mean +- std over {len(losses)} tasks)."
                    )

                    logger.add_scalar(f"{d_name}_loss_mean{t_str}", loss_mean, k)
                    logger.add_scalar(f"{d_name}_loss_std{t_str}", loss_std, k)
                    logger.add_scalar(f"{d_name}_acc_mean{t_str}", acc_mean, k)
                    logger.add_scalar(f"{d_name}_acc_std{t_str}", acc_std, k)
                    logger.add_scalar(f"{d_name}_time_eval{t_str}", time_eval, k)

                    if d_name == "val":
                        logger.add_scalar(
                            "loss", -acc_mean, k
                        )  # running val_loss used for hyperparameter optimization

        if is_final_iter:
            break


class TaskBatch:
    """
    Handles the train and valdation loss for a batch of tasks
    """

    def __init__(
        self,
        reg_param,
        meta_model,
        data,
        task_indexes=None,
        meta_batch_size=None,
        criterion=None,
    ):
        def meta_batch_model_decorator(f):
            def new_f(inp):
                flattened = torch.flatten(inp, start_dim=0, end_dim=1)
                out = f(flattened)
                return torch.reshape(out, shape=[inp.shape[0], inp.shape[1], -1])

            return new_f

        self.meta_model = meta_batch_model_decorator(meta_model)
        self.task_indexes = task_indexes
        self.criterion = criterion

        self.train_input, self.train_target, self.test_input, self.test_target = data

        self.train_shape = [self.train_input.shape[0], self.train_input.shape[1]]
        self.test_shape = [self.test_input.shape[0], self.test_input.shape[1]]

        self.meta_batch_size = (
            meta_batch_size if meta_batch_size is not None else self.train_shape[0]
        )

        self.one_task = True if self.train_input.shape[0] == 1 else False

        self.reg_param = reg_param
        self.train_losses = None
        self.val_losses, self.val_accs = None, None

        self.train_hrepr = self.meta_model(self.train_input)
        self.test_hrepr = self.meta_model(self.test_input)

    # def bias_reg_f(self, bias, params):
    #     # l2 biased regularization
    #     return sum([((b - p) ** 2).sum() for b, p in zip(bias, params)])

    def batch_linear(self, inp, params):
        inp = torch.unsqueeze(inp, dim=2)
        weights, bias = params[0].unsqueeze(dim=1), params[1].unsqueeze(1)
        return (inp @ weights).squeeze(dim=2) + bias

    def batch_loss(self, inp, target):
        input_flat = torch.flatten(inp, start_dim=0, end_dim=1)
        target_flat = torch.flatten(target, start_dim=0, end_dim=1)
        loss = self.criterion(input_flat, target_flat, reduction="none")
        loss = (
            loss.reshape(shape=[inp.shape[0], inp.shape[1], -1]).mean(dim=1).squeeze(1)
        )
        return loss

    def train_loss_f(self, params, hparams=None):
        # biased regularized cross-entropy loss where the bias are the meta-parameters in hparams
        p = (
            [w[self.task_indexes] for w in params]
            if self.task_indexes is not None
            else params
        )
        out = self.batch_linear(self.train_hrepr, p)

        self.train_losses = self.batch_loss(
            out, self.train_target
        ) + 0.5 * self.reg_param * (
            (p[0] ** 2).sum(dim=2).sum(dim=1) + (p[1] ** 2).sum(dim=1)
        )

        return self.train_losses.sum()

    def val_loss_f(self, params, hparams, test=False):
        p = (
            [w[self.task_indexes] for w in params]
            if self.task_indexes is not None
            else params
        )
        # cross-entropy loss (uses only the task-specific weights in params
        out = self.batch_linear(self.test_hrepr, p)
        # out = out.flatten(start_dim=0, end_dim=1)
        # target = self.test_target.flatten(start_dim=0, end_dim=1)

        val_losses = self.batch_loss(out, self.test_target)

        self.val_losses = val_losses.detach().cpu().numpy()  # avoid memory leaks

        pred = out.argmax(
            dim=2, keepdim=True
        )  # get the index of the max log-probability
        self.val_accs = (
            pred.eq(self.test_target.view_as(pred))
            .double()
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        return val_losses.sum() / self.meta_batch_size


def inner_loop(params, inner_optim_build, task, T, log_interval=None):

    inner_opt = inner_optim_build(params)[0]
    for t in range(T):
        task_train_loss = task.train_loss_f(params)
        task_train_loss.backward(inputs=params)
        inner_opt.step()
        inner_opt.zero_grad()

        if log_interval and (t % log_interval == 0 or log_interval == T - 1):
            print(f"t={t} task_train_loss={task_train_loss}")


def evaluate(
    n_tasks,
    dataloader,
    meta_model,
    get_params_init,
    T,
    reg_param,
    inner_optim_build,
    n_parallel_tasks=None,
    criterion=None,
    post_transform=None,
):
    meta_model.eval()
    device = next(meta_model.parameters()).device

    val_losses, val_accs = [], []
    for k, batch in enumerate(dataloader):
        tr_xs, tr_ys = batch["train"][0].to(device), batch["train"][1].to(device)
        tst_xs, tst_ys = batch["test"][0].to(device), batch["test"][1].to(device)
        tr_xs, tr_ys, tst_xs, tst_ys = post_transform([tr_xs, tr_ys, tst_xs, tst_ys])

        if n_parallel_tasks is not None or n_parallel_tasks == -1:
            task_batches = []
            for i in range(int(np.ceil(tr_xs.shape[0] / n_parallel_tasks))):
                end_task_idx = min(n_parallel_tasks * (i + 1), tr_xs.shape[0])
                task_batches.append(
                    [
                        t[i * n_parallel_tasks : end_task_idx]
                        for t in (tr_xs, tr_ys, tst_xs, tst_ys)
                    ]
                )

        else:
            task_batches = [(tr_xs, tr_ys, tst_xs, tst_ys)]

        for (tr_xs, tr_ys, tst_xs, tst_ys) in task_batches:
            params = get_params_init(bs=tr_xs.shape[0])

            task = TaskBatch(
                reg_param,
                meta_model,
                (tr_xs, tr_ys, tst_xs, tst_ys),
                criterion=criterion,
            )

            inner_loop(params, inner_optim_build, task, T)
            task.val_loss_f(params, meta_model.parameters())

            val_losses.extend(task.val_losses)
            val_accs.extend(task.val_accs)

            if len(val_accs) >= n_tasks:
                break

        if len(val_accs) >= n_tasks:
            break

    return np.array(val_losses[:n_tasks]), np.array(val_accs[:n_tasks])


def get_cnn_omniglot(hidden_size, n_classes=None, transductive=False):
    def conv_layer(
        ic,
        oc,
    ):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(
                oc,
                momentum=1.0,
                affine=True,
                track_running_stats=not transductive,  # When this is true is called the "transductive setting"
            ),
        )

    net = nn.Sequential(
        conv_layer(1, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten(),
        # nn.Linear(hidden_size, n_classes)
    )

    initialize(net)
    return net


def get_cnn_miniimagenet(hidden_size, n_classes=None, transductive=False):
    def conv_layer(ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(
                oc,
                momentum=1.0,
                affine=True,
                track_running_stats=transductive,  # When this is true is called the "transductive setting"
            ),
        )

    net = nn.Sequential(
        conv_layer(3, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten(),
        nn.Tanh(),
        # nn.Linear(hidden_size*5*5, n_classes,)
    )

    # initialize(net)
    return net


def set_seed(seed: int, is_deterministic: bool = True):
    # set the seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if is_deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return


def initialize(net):
    # initialize weights properly
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            # m.weight.data.normal_(0, 0.01)
            # m.bias.data = torch.ones(m.bias.data.size())
            m.weight.data.zero_()
            m.bias.data.zero_()

    return net


if __name__ == "__main__":
    main()
