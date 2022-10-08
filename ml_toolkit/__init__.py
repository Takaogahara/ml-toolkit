import os

__version__ = "0.0.1"


def get_root():
    return os.path.abspath(os.path.join(os.path.join(__file__,
                                                     os.pardir), os.pardir))
