from pathlib2 import Path
import os

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"

def ensure_dir(file_path):
    """ create a safely nested folder
    """
    if '.' in os.path.basename(os.path.normpath(file_path)):
        directory = os.path.dirname(file_path)
    else:
        directory = os.path.normpath(file_path)
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except FileExistsError as e:
            # multiprocessing can cause directory creation problems
            print(e)
