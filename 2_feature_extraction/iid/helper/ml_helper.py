import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
sns.set_context("paper", font_scale=1.5)


def model_evaluation(X: pd.DataFrame or np.array, y: np.array or list or pd.Series, clf, params: dict or None,
                     n_repeats: int = 5, n_splits: int = 5, random_state: int = 42, scoring='roc_auc',
                     n_iter=10, n_jobs=-1, sampling=None, refit='balanced_accuracy'):
    if sampling == 'over':
        pipe = imbPipeline(
            [('imputer', KNNImputer(n_neighbors=2)),
             ('sampler', RandomOverSampler(sampling_strategy='minority', random_state=random_state)),
             ('clf', clf)])
    elif sampling == 'under':
        pipe = imbPipeline(
            [('imputer', KNNImputer(n_neighbors=2)),
             ('sampler', RandomUnderSampler(sampling_strategy='majority', random_state=random_state)),
             ('clf', clf)])
    else:
        pipe = Pipeline([('clf', clf)])

    rkf = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=random_state)

    if params is not None:
        rscv = RandomizedSearchCV(pipe, params, random_state=random_state, scoring=scoring, cv=rkf, n_jobs=n_jobs,
                                  n_iter=n_iter, return_train_score=False, refit=refit)
        rscv.fit(X, y)
    else:
        rscv = cross_validate(pipe, X, y, scoring=scoring, cv=rkf, n_jobs=n_jobs)

    return rscv


def drop_correlated_features(X: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 0.95):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X = X.drop(X[to_drop], axis=1)
    X_test = X_test.drop(X_test[to_drop], axis=1)
    print(f'Dropped {len(to_drop)} features', X.shape[1], 'left')
    return X, X_test


def calculate_feature_importance(model, X_test, y_test, n_repeats: int, top_n_features: int, path: str, n_jobs=1):
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42, n_jobs=n_jobs)
    forest_importances_mean = pd.Series(result.importances_mean, index=X_test.columns)
    forest_importances_std = pd.Series(result.importances_std, index=X_test.columns)
    forest_importances = pd.concat([forest_importances_mean, forest_importances_std], axis=1).rename(
        {0: 'mean', 1: 'std'}, axis=1)
    forest_importances = forest_importances.sort_values(by='mean', ascending=False)
    forest_importances.to_csv(f'{path}.csv')
    fig, ax = plt.subplots(dpi=80, figsize=(20, 20))
    y_pos = np.arange(len(forest_importances.iloc[:top_n_features, :].values))
    ax.barh(y_pos, forest_importances.iloc[:top_n_features, 0].values, xerr=forest_importances.iloc[:top_n_features, 1],
            align='center')
    ax.set_yticks(y_pos, labels=forest_importances.iloc[:top_n_features, 0].index)
    ax.invert_yaxis()
    ax.set_title("Feature importances")
    ax.set_ylabel("Mean ROC-AUC decrease")
    fig.tight_layout()
    fig.savefig(f'{path}.png')
    fig.savefig(f'{path}.pdf')


def shap_fi(model, X: pd.DataFrame, X_test: pd.DataFrame, path: str):
    #explainer = shap.TreeExplainer(model)
    explainer=None
    shap_values_train = explainer.shap_values(X)
    shap_values_test = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values_train, feature_names=X.columns, title="SHAP summary plot", show=False,
    #                  plot_size=(10, 10), class_names=model.classes_)
    # plt.tight_layout()
    # plt.savefig(f'{path}_shap_fi.png')
    # plt.savefig(f'{path}_shap_fi.pdf')
    dump(explainer, f'{path}_shap_fi.joblib')

    # plt.show()
    # plt.clf()
    # plt.close()
    return explainer, shap_values_train, shap_values_test


def sub_feature_pcas(df_train_orig: pd.DataFrame, df_test_orig: pd.DataFrame, prefixes: list, n_components: int = 1):
    pca_dfs = []
    pca_test_dfs = []
    for p in prefixes:
        df_train = df_train_orig.loc[:, df_train_orig.columns.str.startswith(p)]
        df_test = df_test_orig.loc[:, df_test_orig.columns.str.startswith(p)]
        pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=n_components))])
        pipe.fit(df_train)
        df_train = pd.DataFrame(pipe.transform(df_train), index=df_train.index, columns=[f'{p}'])
        df_test = pd.DataFrame(pipe.transform(df_test), index=df_test.index, columns=[f'{p}'])
        pca_dfs.append(df_train)
        pca_test_dfs.append(df_test)

        df_train = df_train_orig.loc[:, df_train_orig.columns.str.startswith(f'neighbor_{p}')]
        df_test = df_test_orig.loc[:, df_test_orig.columns.str.startswith(f'neighbor_{p}')]
        pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=n_components))])
        pipe.fit(df_train)
        df_train = pd.DataFrame(pipe.transform(df_train), index=df_train.index, columns=[f'neighbor_{p}'])
        df_test = pd.DataFrame(pipe.transform(df_test), index=df_test.index, columns=[f'neighbor_{p}'])
        pca_dfs.append(df_train)
        pca_test_dfs.append(df_test)

    return pd.concat(pca_dfs, axis=1), pd.concat(pca_test_dfs, axis=1)

