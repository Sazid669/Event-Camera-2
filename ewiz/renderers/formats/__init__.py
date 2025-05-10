from .base import RendererBase
from .video import Renderer2D
# TODO: Option to be added
# from .volume import Renderer3D


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

rendering_formats = {k._name: k for k in inheritors(RendererBase)}
