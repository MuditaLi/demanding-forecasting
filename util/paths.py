import os


def get_root_path(root_dir_name: str, max_parent_directories: int = 10) -> str:
    current_path = os.path.abspath(os.path.dirname(__file__))
    for i in range(max_parent_directories):
        if os.path.basename(current_path) == root_dir_name:
            return current_path
        current_path = os.path.dirname(current_path)
    raise LookupError


ROOT_NAME = 'Carlsberg'
PATH_RESPOSITORY_ROOT = get_root_path(ROOT_NAME)

DIRNAME_DATA = 'data'
DIRNAME_DATA_RAW = 'raw'
DIRNAME_DATA_FORMATTED = 'formatted'
PATH_DATA = os.path.realpath(os.path.join(PATH_RESPOSITORY_ROOT, DIRNAME_DATA))
PATH_DATA_RAW = os.path.realpath(os.path.join(PATH_DATA, DIRNAME_DATA_RAW))
PATH_DATA_FORMATTED = os.path.realpath(os.path.join(PATH_DATA, DIRNAME_DATA_FORMATTED))
PATH_CURRENT_REPOSITORY = os.path.dirname(os.path.dirname(__file__))

# Path to configs file folder
DIRNAME_CONFIG_FILES = 'configs'
PATH_CONFIG = os.path.realpath(os.path.join(PATH_CURRENT_REPOSITORY, DIRNAME_CONFIG_FILES))

# Path to models folder
DIRNAME_MODELS = 'models'
# PATH_MODELS = os.path.realpath(os.path.join(PATH_CURRENT_REPOSITORY, DIRNAME_MODELS))
PATH_MODELS = os.path.realpath(os.path.join(PATH_RESPOSITORY_ROOT, DIRNAME_MODELS))

# Path to outputs folder
DIRNAME_OUTPUTS = 'outputs'
PATH_OUTPUTS = os.path.realpath(os.path.join(PATH_CURRENT_REPOSITORY, DIRNAME_OUTPUTS))
