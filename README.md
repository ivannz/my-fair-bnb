# my-fair-bnb: A toy BnB solver for MILP

How does one learn a new algorithm? By reinventing the wheel.

We implement a simple branch-and-bound search for Mixed Integer Linear Programs with external nodesel and branchrule heuristics and compare it with the powerful SCIP. Our toy bnb solver has neither the primal search heuristics, nor cuts, nor constraint tightening, nor presolving steps. Also, unlike SCIP, the sub-problems are not transformed/simplified in each node, and the search space splitting is done only with respect to the original variable bounds.

The notebook shows examples of random, strong and simple pseudocost branching strategies, and explores the impact of dual-bound oblivious and dual-bound aware depth-first tree traversal.

## Setup

The basic working environment is set up with the following commands:

```bash
# conda deactivate && conda env remove -n toybnb

# pytorch, scip, scipy and other essentials
conda create -n toybnb python pip setuptools numpy "scipy>=1.9" networkx \
  matplotlib scikit-learn notebook "conda-forge::pygraphviz" "conda-forge::pyscipopt" \
  && conda activate toybnb \
  && pip install tqdm ecole

# packages for development and diffs
conda install -n toybnb pytest \
  && pip install "black[jupyter]" pre-commit gitpython nbdime \
  && pre-commit install

# install the package (editable install)
pip install -e .
```
