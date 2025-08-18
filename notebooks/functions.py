'''Helper functions for Jupyter notebooks.'''

import pandas as pd
from sklearn.model_selection import cross_val_score


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