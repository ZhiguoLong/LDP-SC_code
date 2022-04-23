
This is the original implementation of the algorithm LDP-SC in the paper "Clustering based on local density peaks and graph cut" (
[DOI](https://doi.org/10.1016/j.ins.2022.03.091) ;
[Direct Link](https://www.sciencedirect.com/science/article/abs/pii/S0020025522003188)
).


## Requirements

Current environment:

* python == 3.9
* scipy == 1.7.2
* numpy == 1.20.3
* scikit-learn == 1.0.1
* networkx == 2.6.3

## Description

* main.py -- A demo of how to use the LDP-SC algorithm.
* main.ipynb -- Demos of loading some of the used datasets.
* evaluation.py
  * compute_score() -- Get ARI, NMI, ACC metrics.

## Citation

If you find this file useful in your research, please consider citing:

```bibtex
@article{long2022clustering,
  title = {Clustering based on local density peaks and graph cut},
  journal = {Information Sciences},
  volume = {600},
  pages = {263-286},
  year = {2022}
}
```
