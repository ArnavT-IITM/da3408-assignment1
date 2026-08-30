# DA3408 Assignment 1

Experiment management and reproducibility assignment by Arnav Thorat (DA24B027).

## Repository contents

- `writeup.pdf`: one-page write-up covering Questions 1 to 4.
- `q2/q2.py`: MNIST MLP training and six-run MLflow sweep.
- `q2/logging_code.py`: exact MLflow parameter and metric logging excerpt.
- `q2/evidence/`: MLflow comparison screenshots.
- `q3/`: DVC metadata for the two dataset versions.
- `q3/evidence/rollback.png`: proof of rollback from v2 to v1 and back.
- `q4/README.md`: shared Q4 repository link and reproduction summary.
- `q4/evidence/`: Partner A and Partner B MLflow run screenshots.

## Setup

Create and activate a Conda environment, then install the required packages:

```bash
conda create -n da3408-assignment1 python=3.11 -y
conda activate da3408-assignment1
pip install mlflow torch torchvision numpy "dvc[ssh]"
```

## Question 2

Start the MLflow server from the repository root:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

In another terminal, activate the environment and run the six experiments:

```bash
python q2/q2.py
```

Open `http://127.0.0.1:5000` to view and compare the runs. The script varies learning rate over `0.001`, `0.003`, and `0.01`, and batch size over `32` and `128`.

## Question 3

The data was versioned with DVC and stored on an SSH remote during the experiment. The two Git tags are `q3-v1` and `q3-v2`. The rollback was performed with:

```bash
git checkout q3-v1
dvc checkout
wc -l q3/filenames.csv
git checkout main
dvc checkout
wc -l q3/filenames.csv
```

The expected counts are 1801 lines for `q3-v1` and 2801 lines for `q3-v2`, including the CSV header.
The terminal output is included in `q3/evidence/rollback.png`.

## Question 4

The shared Q4 repository is [AakashAadhithya/aiops-assignment1](https://github.com/AakashAadhithya/aiops-assignment1). Aakash completed Partner A's work, and my Partner B reproduction is recorded in commit `95609cc`. See `q4/README.md` for the result and evidence paths.

## AI disclosure

See `AI_DISCLOSURE.md`.
