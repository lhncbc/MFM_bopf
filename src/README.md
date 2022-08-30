# MFM top-level source code directory
Most of the Jupyter Notebooks in this directory are involved either in Extended Data 
Analysis (EDA) or as part of a data field/target generation pipeline.

This is the process for data cleaning / field & target generation:
![Image](MFM_data_creation.png?raw=true)

## Source code subdirectories
The following table summarizes the src subdirectories. They are listed in order of decreasing
importance to the application. The [multi_predict](multi_predict) directory is especially important since it
contains the primary code that was run for the main experimental results. The [imbl](imbl) 
directory is also important since it contains the Neural Network code that was included in 
the final comparison. It also contains the Cramer / Theil feature selection code.

| Directory | Description |
|-----------|-------------|
| multi_predict | Main multi-processing code for running ML algorithms |
| imbl | Contains TensorFlow Imbalanced data and Learned Embedding ANNs |
| shap_lime | Contains code used to determine Shapley value analysis |
| confidence_interval | Code for determining CI for the paper |
| calibrate | Code used to create calibration curves |
| cross_val | Initial cross validation code that eventually was used elsewhere |
| site_compare | Code used to re-run algorithms on a site-specific basis |