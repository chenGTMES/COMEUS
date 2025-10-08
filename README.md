# Framelet Regularized SENSE Calibration in K-Space with Conditional Sensitivity-Map Updates Algorithm (ComeUs)

This repository provides the official implementation of **Framelet Regularized SENSE Calibration in K-Space with Conditional Sensitivity-Map Updates Algorithm**.

## 🔧 Requirements

The environment configuration is provided in [`environment.yml`](./environment.yml).
You can create the environment using:

```bash
conda env create -f environment.yml
```

## 🚀 Getting Started

To run the reconstruction on test data:

```bash
python main.py
```

The code includes a set of **Phantom data** for quick testing.

## 📁 Project Structure

* `main.py` – Entry point for running the reconstruction.
* `gen_mask.m` – MATLAB script for generating sampling masks.
* `environment.yml` – Conda environment specification.
* `data/` – Contains example Phantom test data.
* `mask/` – Contains example undersampled mode.
* `algorithm/ADDL.py` – Our propose ADDL algorithm.
* `utils/` – Supporting modules.

## 📦 Additional test data

Additional test data is available at:

👉 [Google Drive - ADDL Resources](https://drive.google.com/drive/folders/1GkizZg6Qgszza4yq0NN3csjsCeYWi2Mb?usp=drive_link)

* Generate your own sampling patterns via `gen_mask.m`.

* If there have any problem, please send an email to 2019151068@email.szu.edu.cn
