import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mikhail-alien-signals-2226')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///users.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml', 'model.keras')
    MODEL_PATH_H5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml', 'model.h5')
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data (1).npz')
    MAX_CONTENT_LENGTH = 256 * 1024 * 1024

    CLASS_NAMES = [
        '55_Cancri_Bc', 'Gliese_', 'Gliese_12_b', 'Gliese_163_c',
        'HD_20794_d', 'HD_216520_c', 'HIP_38594_b', 'K2-155d',
        'K2-288Bb', 'K2-332b', 'K2-72e', 'Kepler-155c',
        'Kepler-174d', 'Kepler-186f', 'Kepler-22b', 'Kepler-283c',
        'Kepler-296e', 'Kepler-296f', 'Kepler-62e', 'Kepler-62f'
    ]
    NUM_CLASSES = 20
