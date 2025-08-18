'''Helper functions for Jupyter notebooks.'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV


def cross_validate_models(models: dict, training_df: pd.DataFrame, label: str='polarity', cv: int=3) -> dict:
    '''Cross-validate multiple models and return their average accuracy scores.

    Args:
        models (dict): A dictionary where keys are model names and values are model instances.
        training_df (pd.DataFrame): The training DataFrame containing features and labels.
        label (str): The name of the label column in the DataFrame. Default is 'Polarity'.
        cv (int): Number of cross-validation folds. Default is 3.
        random_seed (int): Random seed for reproducibility. Default is 315.

    Returns:
        dict: A dictionary with model names as keys and their average accuracy scores as values.
    '''

    cross_val_scores = {
        'Model': [],
        'Score': []
    }

    for model_name, model in models.items():
        scores = cross_val_score(
            model,
            training_df.drop(label, axis=1),
            training_df[label],
            cv=cv,
            n_jobs=-1
        )

        cross_val_scores['Model'].extend([model_name]*len(scores))
        cross_val_scores['Score'].extend(scores*100)


    return cross_val_scores


def plot_cross_validation(search_results: GridSearchCV, plot_training: bool=False) -> None:
    '''Takes result object from scikit-learn's GridSearchCV(),
    draws plot of hyperparameter set validation score rank vs
    training and validation scores.'''

    results = pd.DataFrame(search_results.cv_results_)
    sorted_results = results.sort_values('rank_test_score')

    plt.title('Hyperparameter optimization')
    plt.xlabel('Hyperparameter set rank')
    plt.ylabel('Validation accuracy (%)')
    plt.gca().invert_xaxis()

    plt.fill_between(
        sorted_results['rank_test_score'],
        sorted_results['mean_test_score']*100 + sorted_results['std_test_score']*100,
        sorted_results['mean_test_score']*100 - sorted_results['std_test_score']*100,
        alpha=0.5
    )

    plt.plot(
        sorted_results['rank_test_score'],
        sorted_results['mean_test_score']*100,
        label='Validation'
    )

    if plot_training:

        plt.fill_between(
            sorted_results['rank_test_score'],
            sorted_results['mean_train_score']*100 + sorted_results['std_train_score']*100,
            sorted_results['mean_train_score']*100 - sorted_results['std_train_score']*100,
            alpha=0.5
        )

        plt.plot(
            sorted_results['rank_test_score'],
            sorted_results['mean_train_score']*100,
            label='Training'
        )

        plt.legend(loc='best', fontsize='small')

    plt.show()