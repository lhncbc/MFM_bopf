### Directory TFIM (TensorFlow Imbalanced)
This Jupyter Notebook from the TensorFlow website was an initial cut at trying
Neural Networks for classification of our MFM CSL data. It worked reasonably well,
especially since it wasn't tuned very much. This implies that it is worth pursuing this
NN solution since it has the potential to improve results with tuning.

There are two options captured here, one with a bias adjustment and one without. There was
very little difference between the two, and I wasn't confident that the bias was actually 
beneficial in our instance, so I broke them up.

### Here is a quick dump of the statistical results:
| Statistic   | Value                |
|-------------|----------------------|
| loss        | 0.48953476548194885  |
| tp          | 1878.0               |
| fp          | 13172.0              |
| tn          | 21842.0              |
| fn          | 191.0                |
| accuracy    | 0.6396461725234985   |
| precision   | 0.12478405237197876  |
| recall      | 0.907684862613678    |
| mcc         | 0.24806705117225647  |
| f2          | 0.05064314976334572  |
| specificity | 0.6245399117469788   |
| ntp         | 503.631591796875     |
| nfn         | 50.73684310913086    |
| nfp         | 3545.5263671875      |
| ntn         | 5898.26318359375     |
| fbeta_score | 0.22806939482688904  |
| auc         | 0.8251489400863647   |
| prc         | 0.19687128067016602  |
| f2          | 0.4025550856458105   |

See the [imbalanced_data_TF-CSL Notebook](imbalanced_data_TF-CSL.ipynb) for more details and plots.
