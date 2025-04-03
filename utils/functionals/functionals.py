import os
from pathlib import Path
import torch
import numpy as np


def IoU(A, B, all_params=False, **kwargs):
    _and = (A.int() & B.int()).sum()
    _or = (A.int() | B.int()).sum()

    if all_params:
        return {'result': _and / _or, 'others': [_and, _or]}
    return _and / _or

def IoX(A, B, tot, all_params=False, **kwargs):
    _and = (A.int() & B.int()).sum()

    if all_params:
        return {'result': _and / tot, 'others': [_and,]}
    return _and / tot

def CosSim(A, B, inter=None, all_params=False, **kwargs):
    if inter is None:
        cossim = (A[None, ...] * B[None, ...]).sum()/torch.max(A.pow(2).sum().sqrt() * B.pow(2).sum().sqrt(), torch.ones((1,1)) * 1e-15)
    else:
        cossim = inter.nonzero().numel()/torch.max(A.pow(2).sum().sqrt() * B.pow(2).sum().sqrt(), torch.ones((1,1)) * 1e-15)
    if all_params:
        return {'result': cossim, 'others': []}
    return cossim