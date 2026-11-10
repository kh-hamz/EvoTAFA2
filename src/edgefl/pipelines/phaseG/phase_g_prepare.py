"""Preparation and clean-only policy calibration, separate from comparison runs."""
from edgefl.experiments.preparation import prepare as execute


def fit_clipping(*args, **kwargs):
    from edgefl.experiments.sessions import fit_clipping as fit
    return fit(*args, **kwargs)


def calibrate(*args, **kwargs):
    from edgefl.experiments.sessions import calibrate as fit
    return fit(*args, **kwargs)
