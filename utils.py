import os
from dotenv import load_dotenv
from typing import List


def list_directories(path: str) -> List[str]:
    """
    List all directories in a given path.

    Args:
        path (str): The path in which to list directories.

    Returns:
        List[str]: The list of directory names.
    """
    return [entry.name for entry in os.scandir(path) if entry.is_dir()]


def list_files(directory: str) -> List[str]:
    """
    List all files in a given directory.

    Args:
        directory (str): The directory in which to list files.

    Returns:
        List[str]: The list of file names.
    """
    return [entry.name for entry in os.scandir(directory) if entry.is_file()]


def load_environment_variables():
    """Load environment variables from .env file."""
    load_dotenv()
    return {
        "embeddings_model_name": os.getenv("EMBEDDINGS_MODEL_NAME"),
        "persist_directory": os.getenv('PERSIST_DIRECTORY'),
        "model_n_ctx": os.getenv('MODEL_N_CTX'),
        "models_path": os.getenv('MODELS_PATH'),
        "target_source_chunks": int(os.getenv('TARGET_SOURCE_CHUNKS', 4))
    }


def list_all(path):
    contents = {}
    for directory in list_directories(path):
        directory_path = os.path.join(path, directory)
        contents[directory] = list_files(directory_path)
    return contents
