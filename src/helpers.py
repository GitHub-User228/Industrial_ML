import os
import pickle

def get_project_dir():
    return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def save_model(model, filename):
    path = os.path.join(get_project_dir(), f'models/{filename}')
    pickle.dump(model, open(path, "wb"))

def load_model(filename):
    path = os.path.join(get_project_dir(), f'models/{filename}')
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model





