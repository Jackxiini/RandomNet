# RandomNet: Clustering Time Series Using Untrained Deep Neural Networks
Fast and accurate time series clustering method.

## This [[paper]]() is accepted by DMKD!

## Prerequisites

- Python 3.8
- NumPy
- TensorFlow 2.1
- PyMetis
- Scikit-learn
- Linux system

## Installation

1. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Model

To run the model on the Coffee dataset:
```sh
python RandomNet.py --dataset Coffee

Output:
--------------------
Working on dataset:  Coffee  ...
dataset: Coffee
Rand Index: 1.000000
time_cost: 29.610617
