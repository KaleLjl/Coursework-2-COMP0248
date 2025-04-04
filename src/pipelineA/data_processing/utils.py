import pickle
import os

def load_pickle_labels(file_path):
    """Loads labels from a pickle file (.dat).

    Args:
        file_path (str or Path): The path to the pickle file.

    Returns:
        object: The data loaded from the pickle file, or None if the file
                does not exist or an error occurs during loading.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Label file not found at {file_path}")
        return None
        
    try:
        with open(file_path, 'rb') as f:
            labels = pickle.load(f)
        return labels
    except Exception as e:
        print(f"Error loading pickle file {file_path}: {e}")
        return None
