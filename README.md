# my-fair-bnb: A toy BnB solver for MILP

How does one learn a new algorithm? By reinventing the wheel.

We implement a simple branch-and-bound search for Mixed Integer Linear Programs with external nodesel and branchrule heuristics and compare it with the powerful SCIP. Our toy bnb solver has neither the primal search heuristics, nor cuts, nor constraint tightening, nor presolving steps. Also, unlike SCIP, the sub-problems are not transformed/simplified in each node, and the search space splitting is done only with respect to the original variable bounds.

The notebook shows examples of random, strong and simple pseudocost branching strategies, and explores the impact of dual-bound oblivious and dual-bound aware depth-first tree traversal.

## Setup

The basic working environment is set up with the following commands:

```bash
# ensure micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# setup the developer's env including pytorch, scip, scipy and other essentials
# XXX '=' fuzzy prefix version match, '==' exact version match
# XXX micromamba deactivate && micromamba env remove -n toybnb
micromamba create -n toybnb \
  "python=3.11"             \
  numpy                     \
  "scipy>=1.9"              \
  networkx                  \
  "conda-forge::pyscipopt"  \
  scikit-learn              \
  pytorch                   \
  torch-scatter             \
  ecole                     \
  einops                    \
  matplotlib                \
  jupyter                   \
  plotly                    \
  "conda-forge::pygraphviz" \
  tqdm                      \
  pydantic                  \
  "black[jupyter]"          \
  pre-commit                \
  gitpython                 \
  nbdime

# install the package (editable install)
micromamba run -n toybnb pip install -e .

# A custom fork of ecole with extra obs (Tree-MDP and nodesel)
# pip install -vv "ecole @ git+https://github.com/ivannz/ecole.git"
pre-commit install
```
