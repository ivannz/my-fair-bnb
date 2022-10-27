# A toy BnB solver for MILP

How does one learn a new algorithm? By reinventing the wheel.

## Setup

The basic working environment is set up with the following commands:

```bash
# conda deactivate && conda env remove -n toybnb

# pytorch, scip, scipy and other essentials
conda create -n toybnb python pip setuptools numpy "scipy>=1.9" networkx \
  matplotlib notebook \
  && conda activate toybnb \
  && pip install tqdm

# packages for development and diffs
conda install -n toybnb pytest \
  && pip install "black[jupyter]" "conda-forge::pyscipopt" pre-commit gitpython nbdime \
  && pre-commit install

# install the package (editable install)
pip install -e .
```
