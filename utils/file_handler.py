import pandas as pd
import logging
import os
import pickle
import hashlib

PROJECT_ROOT = os.path.abspath("..")
PROJECT_NAME = os.path.basename(PROJECT_ROOT)
DATASETS_PATH = f'{PROJECT_ROOT}/datasets'
MODELS_PATH = f'{PROJECT_ROOT}/models'
DF_PATH = f'{PROJECT_ROOT}/dataframes'
IMAGES_PATH = f'{PROJECT_ROOT}/images'
GITIGNORE_PATH = f'{PROJECT_ROOT}/.gitignore'

GITHUB_FILE_LIMIT_BYTES = 100 * 1024 * 1024  # 100 MB

def hash_object(obj):
    """Generate hash of a pickled object in memory."""
    return hashlib.sha256(pickle.dumps(obj)).hexdigest()

def save_model_pickle(model, filename:str) -> None:
    """
    To save model to a file run this in notebook cell:
    save_model_pickle(model=your_model_variable, filename='your_file_name_no_extension')
    # Example:
    save_model_pickle(model=final_linear_regression_model, filename='final_linear_regression_model')
    """

    model_file_path = f'{MODELS_PATH}/{filename}.pkl'

    # Ensure model directory exists
    os.makedirs(MODELS_PATH, exist_ok=True)

    new_model_hash = hash_object(model)

    # Check if the file already exists and compare content hashes
    if os.path.exists(model_file_path):
        try:
            with open(model_file_path, "rb") as f:
                existing_model = pickle.load(f)
            existing_model_hash = hash_object(existing_model)
            if new_model_hash == existing_model_hash:
                logging.info("No changes detected in model. Skipping save.")
                return
        except Exception as e:
            logging.warning(f"Could not load existing model for comparison: {e}")

    # Save the updated model
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)

    logging.info(f"Model file pickle is updated: {model_file_path}")

    # Check file size and update .gitignore if needed
    file_size = os.path.getsize(model_file_path)
    if file_size > GITHUB_FILE_LIMIT_BYTES:
        logging.warning(f"{model_file_path} exceeds GitHub size limit ({file_size} bytes)")
        # Append to .gitignore if not already present
        relative_path = os.path.relpath(model_file_path, PROJECT_ROOT)
        if os.path.exists(GITIGNORE_PATH):
            with open(GITIGNORE_PATH, "r") as g:
                ignored_files = g.read().splitlines()
        else:
            ignored_files = []

        if relative_path not in ignored_files:
            with open(GITIGNORE_PATH, "a") as g:
                g.write(f"\n{relative_path}")
            logging.info(f"Added to .gitignore: {relative_path}")


def save_df_pickle(df: pd.DataFrame, filename:str):
        df_file_path = f'{DF_PATH}/{filename}.pkl'
        df.to_pickle(df_file_path)
        logging.info(f"Backup file is created: {df_file_path}")

def read_df_pickle(local_path) -> pd.DataFrame:
    df_file_path = f'{PROJECT_ROOT}/{local_path}'
    return pd.read_pickle(df_file_path)