## multi-predict directory
This directory name is historical in that the initial goal of the code was
test out multiprocessing on the compute nodes. This was done primarily to 
do a hyperparameter grid search, while outputing a large amount of 
statistical data.

This table gives a quick description of the Python programs that are designed
to run on a compute server with multiple processors. (Note: parallelism can 
only be done in Python with multiprocessing, not multithreading, because of
the Python global interpreter lock (GIL))

## Python code
| Python file     | Description |
|-----------------|-------------|
| predict_main.py | Main program file that accepts numerous command line arguments that control the running of the machine learning algorithms on the compute node. Can perform under or oversampling using various techniques and run a parameter grid search using the "pred_params" variable. It calls routines in predict_out.py to create an output file per run.|
| predict_out.py  | Primarily contains the "save_to_file()" routine which outputs a large number of statistical files per each set of hyperparameters. |

## Jupyter Notebooks
| Notebook | Description|
|----------|------------|
|ComparAllResults.ipynb | Large Jupyter notebook that compares the performance of the different machine learning algorithms against each other. Current saved file out-of-date, but is representative of final results.  
| spread_coeffs_combo.ipynb | Analysis focuses on the algorithm coefficients to estimate the relative predictive power of each feature. |

## Scripts
| Script | Description|
|--------|------------|
| slurmMultiPred.sh | Script used to run predict_main via Slurm on a compute server.|
| multiPredTmplt.sh | Script that allows parameterization of the hyperparameters via an environment variable |
