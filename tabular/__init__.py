"""
Package for tabular data.
Adapted from https://github.com/gpapamak/maf/tree/master/datasets
"""

from .hepmass import HEPMASS
from .human import HUMAN
from .crop import CROP
from .utils import *

TAB_DSETS = {"HEPMASS": HEPMASS,
             "HUMAN": HUMAN,
             "CROP": CROP
             }
