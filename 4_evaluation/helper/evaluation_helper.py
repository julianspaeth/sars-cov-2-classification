import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import cross_validate

from sklearn.preprocessing import StandardScaler


def hyperparameter_search(features: pd.DataFrame, labels: pd.DataFrame, scoring="balanced_accuracy", random_state=1,
                          n_estimators=200, n_iter=100):
    n_repeats = 5
    n_splits = 5
    n_jobs = -1

    rkf = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=random_state)
    params = {
        'clf__max_depth': [None, 2, 5, 10, 20, 30, 40, 50, 70, 100],
        'clf__min_samples_leaf': [1, 3, 5, 10, 15, 20, 25, 30],
        'clf__min_samples_split': [2, 3, 5, 10, 15, 20, 25, 30],
        'clf__max_features': [None, 'sqrt', "log2"],
        'clf__bootstrap': [True, False],
        'clf__class_weight': [None, 'balanced', 'balanced_subsample'],
        'imputer__n_neighbors': [1, 3, 5]
    }
    pipe = Pipeline(
        [('scaler', StandardScaler()),
         ('imputer', KNNImputer()),
         ('clf', RandomForestClassifier(random_state=random_state, n_estimators=n_estimators, n_jobs=n_jobs))])

    X, X_test, y, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, stratify=labels,
                                            random_state=random_state)
    X, y = RandomUnderSampler(random_state=random_state, sampling_strategy='not minority').fit_resample(X, y)

    rs_result = RandomizedSearchCV(pipe, params, random_state=random_state, n_iter=n_iter, scoring=scoring, cv=rkf,
                                   refit=True, n_jobs=n_jobs)
    rs_result.fit(X, y)
    rs_results_cv = pd.DataFrame(rs_result.cv_results_)
    mask = [x for x in rs_results_cv.columns if "split" in x and "param" not in x]
    rs_results_cv['min_test_score'] = rs_results_cv.loc[:, mask].min(axis=1)
    rs_results_cv['max_test_score'] = rs_results_cv.loc[:, mask].max(axis=1)
    rs_result_df = rs_results_cv.loc[:,
                   ['params', 'mean_test_score', 'std_test_score', 'min_test_score', 'max_test_score']].sort_values(
        'mean_test_score', ascending=False)

    cv_result = pd.DataFrame.from_dict(cross_validate(rs_result.best_estimator_, X, y, cv=rkf, scoring=scoring))

    model = rs_result.best_estimator_.fit(X, y)

    return rs_result_df, cv_result, model, X, y, X_test, y_test


def cv_evaluate(features: pd.DataFrame, labels: pd.DataFrame, scoring="balanced_accuracy", random_state=1,
                n_estimators=200, sampling=None):
    n_repeats = 5
    n_splits = 5
    n_jobs = -1

    rkf = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=random_state)
    if sampling is None:
        pipe = Pipeline(
            [('scaler', StandardScaler()),
             ('imputer', KNNImputer()),
             ('clf',
              BalancedRandomForestClassifier(random_state=random_state, n_estimators=n_estimators, n_jobs=n_jobs))])
    else:
        pipe = Pipeline(
            [('scaler', StandardScaler()),
             ('imputer', KNNImputer()),
             ('sampler', sampling),
             ('clf',
              BalancedRandomForestClassifier(random_state=random_state, n_estimators=n_estimators, n_jobs=n_jobs))])

    X, X_test, y, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, stratify=labels,
                                            random_state=random_state)
    # X, y = RandomUnderSampler(random_state=random_state, sampling_strategy='not minority').fit_resample(X, y)

    rs_result = cross_validate(pipe, X, y, cv=rkf, scoring=scoring, n_jobs=n_jobs)
    rs_results_cv = pd.DataFrame(rs_result)

    model = pipe.fit(X, y)

    return rs_results_cv, model, X, y, X_test, y_test
