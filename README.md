# pysme-wrapper

## Installation

### 1. Install Miniforge (recommended)

It is recommended to use Miniforge for managing your Python environment:

https://conda-forge.org/download/

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

