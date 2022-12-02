# P5-Nonlinear-Dimensionality-Reduction
5th semester project concerning feature engineering and nonlinear dimensionality reduction in particular.

## Running the script
The script has two parameters:
1. The number of elements from MNIST to run linear dimensionality reduction with (default 60000) 
2. The number of elements from MNIST to run non linear dimensionality reduction with (default 15000 or parameter 1).

The following command from the root repo directory will run the script with the first 1000 data elements in MNIST for both linear and non linear methods, and save the gridsearch log to logfile.txt.

```shell
python src/pipeline.py 1000 | tee -i src/results/logfile.txt
```

Omit the tee part to print to the console as first priority

```shell
python src/pipeline.py 1000
```