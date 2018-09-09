# Data Science London + Scikit-learn

![Data Science London + Scikit-learn](https://kaggle2.blob.core.windows.net/competitions/kaggle/3428/media/scikit-learn-logo.png)

This is a project that has long ended four years ago. However, it still proves to be one of the most useful project for machine learning and data science practitioners to hone their skills with Scikit-Learn.

## Instructions

### Environment
Please notice that the following libraries are needed to run the Python file:
```
numpy pandas sklearn
```

If you are using `Anaconda`, you can create a new environment and install these packages
```
conda create -n kaggle-sklearn-london numpy pandas sklearn
```

### Running

The file can be run with the following code:
```
python london-scikit-learn.py
```

*Note*: The data file could not be downloaded with the code, thus you must acquire the 3 csv files from [the project page](https://www.kaggle.com/c/data-science-london-scikit-learn).

## Credits

With my customized ensemble model, I can reach the score around 0.86. Not until I learnt about Gaussian Mixture models from (Siddharth Agarwal's repository)[https://kaggle2.blob.core.windows.net/competitions/kaggle/3428/media/scikit-learn-logo.png] can I reach the score above 0.98, which is really a breakthrough.

## About Gaussian Mixture

For practitioners, I would recommend [Scikit-Learn's Introduction about Gaussian Mixture Models](http://scikit-learn.org/stable/modules/mixture.html) and [the documentation on it](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).

It turns out that this model is extremely useful for classification tasks, playing a very efficient role during the preprocessing phase.

For your convenience of reading my code, special attention is worth being paid to GaussianMixture's methods of [`aic(x)`](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html), [`bic(x)`](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.bic), and [`fit(X, y=None)`](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.bic).

The parameters of `n_components` and `covariance_type` are also worth special notice.
