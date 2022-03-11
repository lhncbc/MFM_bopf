The original dataset of 780 variable  and 224438 observations was reviewed for variable selection.
Only maternal variables were retained.
We selected only patients with onepregflag = 1
Continous variables like maternal age, height, BMI, Blood loss, were transformed to categorical (see Rstats file notes for values)
In continous variables with normal distribution missing and unknown variables were replace with value of the median
On the other variables  missing and unknown values lumped together in one because it does not make any difference the end result is we don't know the value.

Outcome variable for first experiments = transfus_yes
created by combining : Bloodproduct and Posttransfus

We have 248 working variables including MomID and Sitenum. MomID is id variable Sitenum might or might not be a predictor variable.

Plan:
1. Correlation Coefficient
2. Principal Components
3. Discuss results of Correlation Coefficient and Principal Components to select predictor varialbles 
3. Laritza to create similar demographics cohorts
4. Mike to test undersampling and oversampling techniques 
e.g. here:

https://towardsdatascience.com/sampling-techniques-for-extremely-imbalanced-data-part-i-under-sampling-a8dbc3d8d6d8

https://towardsdatascience.com/sampling-techniques-for-extremely-imbalanced-data-part-ii-over-sampling-d61b43bc4879

https://imbalanced-learn.readthedocs.io/en/stable/under_sampling.html