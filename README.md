# GFlowNet-CombOpt
Pytorch implementation for our paper 

[Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets](https://arxiv.org/abs/2305.17010).

[Dinghuai Zhang](https://zdhnarsil.github.io/), [Hanjun Dai](https://hanjun-dai.github.io/), Nikolay Malkin, Aaron Courville, [Yoshua Bengio](https://yoshuabengio.org/), [Ling Pan](https://ling-pan.github.io/).

<!-- <p align="center"> -->
<img src="https://s1.ax1x.com/2023/05/30/p9jE7P1.png" border="0" width=60% class="center" />
<!-- </p> -->

We formulate a set of graph combinatorial optimization problems as sequential decision-making sampling problems,
and design efficient GFlowNet algorithms to tackle them.

## For ME


### GPU

ubuntu20.04-cuda11.3.0-py38-torch1.11.0-tf1.15.5-1.8.1

https://pypi.tuna.tsinghua.edu.cn/simple/dgl/

```
pip3 install networkx==2.5.0 pydot hydra-core==1.1.0 omegaconf fairseq submitit hydra-submitit-launcher
pip3 uninstall dgl -y
pip3 install dgl -f https://pypi.tuna.tsinghua.edu.cn/packages/a3/a7/30b040017feb4c44e22f28f858cc878da7458cf8362aec1fc9df7c701e8f/dgl-0.6.1-cp36-cp36m-manylinux1_x86_64.whl
pip3 install ipdb einops
```

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
fairseq 0.12.2 requires hydra-core<1.1,>=1.0.7, but you have hydra-core 1.1.0 which is incompatible.
fairseq 0.12.2 requires omegaconf<2.1, but you have omegaconf 2.1.2 which is incompatible.
```

```
cd GFlowNet-CombOpt/
cd data/
python3 rbgraph_generator.py --num_graph 4000 --save_dir rb200-300/train
python3 rbgraph_generator.py --num_graph 500 --save_dir rb200-300/test
cd ..
cd gflownet/
python3 main.py input=rb200-300 alg=fl bsit=8 > log
```


## Dependency

```bash
pip install hydra-core==1.1.0 omegaconf submitit hydra-submitit-launcher
pip install dgl==0.6.1
```

## Data generation

```bash
cd data/
python rbgraph_generator.py --num_graph 4000 --save_dir rb200-300/train
python rbgraph_generator.py --num_graph 500 --save_dir rb200-300/test  
```

## Training

```bash
cd gflownet/
python main.py input=rb200-300 alg=fl bsit=8
```
