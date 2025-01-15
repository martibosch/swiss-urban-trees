[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/martibosch/swiss-urban-trees/main.svg)](https://results.pre-commit.ci/latest/github/martibosch/swiss-urban-trees/main)
[![GitHub license](https://img.shields.io/github/license/martibosch/swiss-urban-trees.svg)](https://github.com/martibosch/swiss-urban-trees/blob/main/LICENSE)

# Swiss urban tree inventory

Urban tree crown detection and species identification from aerial imagery in Switzerland using deep learning.

## Instructions

1. Create a conda environment:

```bash
snakemake -c1 create_environment
```

2. Activate it (if using conda, replace `mamba` for `conda`):

```bash
mamba activate swiss-urban-trees
```

3. Register the IPython kernel for Jupyter:

```bash
snakemake -c1 register_ipykernel
```

## Acknowledgments

- Based on the [cookiecutter-data-snake :snake:](https://github.com/martibosch/cookiecutter-data-snake) template for reproducible data science.
