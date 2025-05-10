"""
Initialization of all loss functions.

Adapted from 'Secrets of Event-based Optical Flow':
https://github.com/tub-rip/event_based_optical_flow/blob/main/src/costs/__init__.py
"""

from .base import LossBase

# Image variance loss functions
from .variance.variance import ImageVariance
from .variance.normalized import NormalizedImageVariance
from .variance.multifocal import MultifocalNormalizedImageVariance

# Gradient magnitude loss functions
from .gradient.gradient import GradientMagnitude
from .gradient.normalized import NormalizedGradientMagnitude
from .gradient.multifocal import MultifocalNormalizedGradientMagnitude

# Regularizer function
from .regularizer.regularizer import Regularizer


def inheritors(object):
    subclasses = set()
    parents = [object]
    while parents:
        parent = parents.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                parents.append(child)
    return subclasses

loss_functions = {k.name: k for k in inheritors(LossBase)}


from .hybrid import LossHybrid
