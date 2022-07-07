# RandomUnitaryLayers

## Installation
1. Install rust on your system: https://www.rust-lang.org/learn/get-started
2. Install OpenBLAS using your favorite package manager (homebrew for OSX, apt for Ubuntu, ...)

3. Prepare your python environment by installing `maturin`, `numpy`, `wheel`, and upgrading `pip`:
   1. `> pip install maturin numpy wheel`
   2. `> pip install --upgrade pip`
4. Clone the repository:
   1. `> git clone git@github.com:Renmusxd/SimpleUnitaryBuilder.git`
5. Run `make` in the parent directory
   1. `> make`
6. Install the resulting wheel with pip
   1. `> pip install target/wheels/*`
   2. If multiple versions exist, select the one for your current python version.

See [notebook](https://github.com/Renmusxd/RandomUnitaryLayers/blob/main/jupyter/Test.ipynb) for example usage.
