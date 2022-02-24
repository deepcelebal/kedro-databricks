import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import RandomizedSearchCV


def oversample_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """
    Oversample current data with SMOTE.

    Args:
        data: data for the modeling
    
    Returns:
        Oversampled X and y data.
    """

    X = data[parameters['features']]
    y = data.stroke

    sm = SMOTE(random_state=parameters['random_state'])
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res
    


def split_data(X_res: pd.DataFrame, y_res: pd.Series, parameters: Dict) -> Tuple:
    """
    Splits data into train and test splits.
    Args:
        data: DataFrame containing the features and target to split
        parameters: Dictionary of parameters to split the data using sklearn.train_test_split
    Returns:
        Tuple containing the splited data.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X_res,
        y_res,
        test_size = parameters['test_size'],
        random_state = parameters['random_state'],
        stratify=y_res
    )

    return X_train, X_test, y_train, y_test





def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict) -> RandomizedSearchCV:
    """
    Use Randomized Search to tune the Gradient Boost hyperparameters and train model

    Args:
        X_train, y_train : training data
    Return:
        RandomizedSearchCV with the model parameters.
    """


    #oversample data
    sm = SMOTE(random_state=parameters['random_state'])
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    params = {
        'learning_rate': [0.05, 0.01, 0.0001],
        'n_estimators': [90, 140, 200],
        'min_samples_split': [1, 2, 3, 4],
        'max_depth' : [1, 2, 3, 4],
        'warm_start': [False, True]
    }

    clf = GradientBoostingClassifier()
    RSVC = RandomizedSearchCV(
        clf,
        params,
        verbose=3,
        cv=10,
        n_jobs=-1,
        n_iter=10,
        scoring = 'f1'
    )

    RSVC.fit(X_train_res, y_train_res)
    return RSVC



def evaluate_model(classifier: RandomizedSearchCV, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Calculates and logs the f1 score and recall.
    Args:
        classifier: Trained LGBMClassifier
        X_test, y_test: Test_data for evaluation
    """

    y_pred = classifier.predict(X_test)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a f1_score of %.3f and a recall of %3f.", f1, recall)