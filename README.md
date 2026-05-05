# pysme-wrapper

## Installation

### 1. Install Miniforge (recommended)

It is recommended to use Miniforge for managing your Python environment:
https://conda-forge.org/download/

Alternatively, you can also use conda, miniconda or any other environment manager.

### 2. Create and activate the environment

Use the provided environment file. You can edit it to include more packages or change the environment name.
```bash
mamba create -f environment.yaml
mamba activate <environment_name>
```

If you choose not to use conda, ensure that **all packages listed in environment.yaml** are installed by other means, as they are required dependencies.

### 3. Install pip packages

```bash
pip install PyAstronomy pysme-astro
```

### 4. Install this package

From the root directory of this repository, run:
```bash
pip install -e .
```
This installs the package in editable mode, so any changes to the source code are immediately reflected without reinstalling.

## Tutorial

For the typical workflow used with pymse-wrapper, see the pysme_wrapper_test.ipynb notebook.

## PySME documentation

For PySME documentation, see https://pysme-astro.readthedocs.io/en/stable/. pysme-wrapper creates helper classes and functions to streamline workflow based on my experience with noisy FEROS spectra, so it is quite possible you will want to use "pure" pysme instead of the wrapper at some point.