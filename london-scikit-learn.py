 #!/usr/bin/env python

import numpy as np
import matplotlib
import pandas as pd

import time
import os

from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

# Import data

def import_data(dir="./", train_features_file="train.csv", train_labels_file="trainLabels.csv", test_features_file="test.csv", header=None):
    """
    Import 3 CSV data files with Pandas' read_csv function:
    - train.csv: training features.
    - trainLabels.csv: training labels.
    - test.csv: testing features.

    The imported data would be further transformed into
    Numpy multi-dimensional arrays for the convenience of
    feeding into other functions.

    Please be aware that the three CSV files come without headers,
    hence the parameter 'header=None'.

    Returns:
    Numpy ndarrays of training features, training labels, 
        and testing features.
    """
    train_features = pd.read_csv(train_features_file, header=header)
    train_labels = pd.read_csv(train_labels_file, header=header)
    test_features = pd.read_csv(test_features_file, header=header)

    X_train = np.asarray(train_features)
    Y_train = np.asarray(train_labels)
    Y_train = Y_train.ravel() # For Grid Searching

    X_test = np.asarray(test_features)

    assert(X_train.shape[0] == Y_train.shape[0])

    return X_train, Y_train, X_test

## Preprocessing Data ##

def concatenate_and_preprocessing(X_train, X_test):
    """
    The most important step, Gaussian Mixture Model, is
    applied within this function.

    Params:
    - X_train, X_test: Numpy ndarray. Untrained features.

    Returns:
    - X_train, X_test: Numpy ndarray. GaussianMixture transformed features.
    """
    # Concatenate all feature data, including those of the training
    # testing set, for the convenience of transformation with
    # trained GaussianMixture class.
    X_all = np.r_[X_train, X_test]
    assert (X_all.shape[0] == X_train.shape[0] + X_test.shape[0])

    ## Gaussian Mixture Model

    # Bayesian information criterion: the lower the better.
    bic = []
    lowest_bic = np.inf

    # Customized "Grid Searching" due to the absence of bic
    # in sklearn.model_selection.GridSearchCV
    n_components_range = range(1, 7) # Similar to 'K' in K Means
    cv_types = ['spherical', 'tied', 'diag', 'full'] # Covariance Types

    # Loop
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Initialize a new GM Model and fit
            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)
            gmm.fit(X_all)
            bic.append(gmm.aic(X_all))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm # The lower the better.
    
    best_gmm.fit(X_all)
    print ("Best Gaussian Misture Model:\n{}".format(best_gmm))
    
    X_train = best_gmm.predict_proba(X_train)
    X_test = best_gmm.predict_proba(X_test)

    return X_train, X_test

## Training ##

def customized_ensemble(classifiers=[
    SVC(C=4, degree=1, gamma=.12), 
    GradientBoostingClassifier(learning_rate=.1), 
    RandomForestClassifier(criterion='entropy'), 
    DecisionTreeClassifier(criterion='entropy'),
    ],
    data=None, labels=None, threshold=.5, training=False,
    trained_classifiers=None):
    """
    Params:
    - data: training set features that have been processed.
    - labels: training labels with the shape of (-1,), if training.
    
    Return:
    - List of estimators, if training.
    - Predictions, with the shape of (-1,).
    - Accuracy Score, if training.
    """
    if training:
        clf_lst = classifiers
    else:
        clf_lst = trained_classifiers
    
    raw_preds = np.zeros(shape=(len(clf_lst), data.shape[0]))

    # Training/Predicting
    for i, clf in enumerate(clf_lst):
        if training:
            clf.fit(data, labels)
        raw_preds[i] = clf.predict(data)
        
    # Voting
    preds = (raw_preds.mean(axis=0) >= threshold) * 1
    
    if not training:
        return preds
    else:
        assert (preds.shape == labels.shape)
        score = (preds == labels).sum() / preds.shape[0]
        print ("Training Score(Accuracy): {}".format(score))
        return clf_lst, preds, score

def best_clf_pred(X_test, trained_classifiers, output_path='./submissions', output=True):
    """
    Application of the best ensemble estimators to the testing features.

    Params:
    - X_test: Numpy ndarray; testing features. Should have been transformed by
        GaussianMixture.
    - trained_classifiers: classifiers returned by the 'customized_ensemble' function after
        training is done.

    Returns:
    - file_name, result: The name of the file outputed to the current directory, if
        the parameter 'output' is True, which is set default. 'result', the Pandas
        DataFrame containing the predictions on the testing set in the required format,
        is returned as well.
    - pred: if output is explicitly set to be False, then only predictions are returned.
        No files would be created.
    """
    pred = customized_ensemble(trained_classifiers=trained_classifiers, data=X_test)

    if output:
        if not os.path.exists(output_path):
        	   os.makedirs('submissions')
        result = pd.DataFrame(pred, columns=['Solution'], index=range(1, len(pred)+1))
        result.index.name = 'Id'

        file_name = time.strftime("%Y_%m_%d-%H:%M", time.localtime())

        # Output to CSV File
        result.to_csv(os.path.join(output_path, file_name))
        print ("File exported as {}".format(os.path.join(output_path, file_name)))

        return file_name, result

    return pred

# Unit Test
if __name__ == '__main__':
    X_train, Y_train, X_test = import_data()
    X_train, X_test = concatenate_and_preprocessing(X_train, X_test)
    
    # Training
    clfs, _, score = customized_ensemble(data=X_train, labels=Y_train, training=True)

    # Predicting
    file_name, _ = best_clf_pred(X_test, trained_classifiers=clfs, output=True)

    assert (os.path.isfile(os.path.join("./submissions/", file_name))) # Make sure the file is created in the target directory