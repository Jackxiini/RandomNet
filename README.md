# RandomNet: Clustering Time Series Using Untrained Deep Neural Networks
RandomNet is a highly accurate and efficient time series clustering method. It has linear time complexity w.r.t the number of instances and the time series length. 

![Overall Architecture](overview.pdf)

## This [[paper]]() is accepted by DMKD!

### Prerequisites
- Python 3.8
- NumPy
- TensorFlow 2.1
- PyMetis
- Scikit-learn
- Linux system

### Installation
1. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
### Dataset
We use the UCR Time Series Classification Archive. You can download the full UCR datasets from [[here]](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).

### Running the Model
To run the model on the Coffee dataset:
```sh
python RandomNet.py --dataset Coffee

Output:
--------------------
Working on dataset:  Coffee  ...
dataset: Coffee
Rand Index: 1.000000
time_cost: 29.610617
