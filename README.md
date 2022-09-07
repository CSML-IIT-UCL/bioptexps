# BioptExps

Bilevel optimization experiments of the paper 
_[Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start](https://arxiv.org/abs/2202.03397)_.


## Getting Started
1. **Install** the packages in [requirements.txt](requirements.txt). The packages version is the one used for the results in the paper, although others could be used. We suggest to use a GPU to speed up the computation. For the random search install the experiment manager [Guild](https://github.com/guildai).
2. **Run** one of the following files (Optionally modify the arguments in each file):
   - [DEQs.py](./source/DEQs.py) for the experiments with equilibrium models.
   - [meta_learning_parallel.py](./source/meta_learning_parallel.py) for the meta-learning experiments.   
   - [poisoning](./source/poisoning.py) for the experiments on data poisoning adversarial attack .
   - [poisoning_random_search.py](.source/poisoning_random_search.py) for the random search on the data poisoning experiments (Uses [Guild](https://github.com/guildai)).

## Additional Info
[hypergrad.py](./source/hypergrad.py) contains the AID hypergradient (i.e. the gradient of the bilevel objective) approximation method which relies on `torch.optim.Optimizers` to solve the linear system. The method is taken from [hypertorch/hypergrad/hypergradients.py](https://github.com/prolearner/hypertorch/blob/master/hypergrad/hypergradients.py).
See [hypertorch](https://github.com/prolearner/hypertorch) for more details on hypergradient approximation methdos and some quick examples on how to incorporate them in a project.

the class`TorchBiOptimizer` in [bilevel_optimizers.py](./source/bilevel_optimizers.py#L15) allows to implement different bilevel optimization methods by varying its parameters. These parameters are specified for several bilevel methods in [guild.ylm](./source/guild.yml).

[data.py](./source/data.py) and [utils.py](./source/utils.py) contain the loading function for the datasets and some utility functions respectively.

Details on the experimental settings can be found in [the paper](https://arxiv.org/abs/2202.03397). Note that the keywords _upper-level_ and _lower-level_ are replaced by `outer` and `inner` respectively in the code.

## Cite Us
If you use this code, please cite [our paper](https://arxiv.org/abs/2202.03397).

```
@article{grazzi2022bilevel,
  title={Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start},
  author={Grazzi, Riccardo and Pontil, Massimiliano and Salzo, Saverio},
  journal={arXiv preprint arXiv:2202.03397},
  year={2022}
}
```