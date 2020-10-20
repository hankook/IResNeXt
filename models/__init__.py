from .resnext import *
from .anytime_resnext import *

def get_model(name, args):
    model = globals()[name]
    return model(**vars(args))
