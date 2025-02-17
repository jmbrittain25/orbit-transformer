import os
import numpy as np


def make_dirs(*dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def check_vals(func=None, val_lower_limit=None, val_upper_limit=None):
    # If the decorator is used without arguments
    if func is None:
        # Return a decorator with the given limits
        def decorator(func):
            return check_vals(func, val_lower_limit, val_upper_limit)
        return decorator

    def wrapper(val, min_val, max_val):
        # normalizing (check val within min/max)
        if val_lower_limit is None or val_upper_limit is None:
            if min_val > max_val:
                raise ValueError(f'min_val ({min_val}) is greater than max_val ({max_val})')
            if np.any(val < min_val) or np.any(val > max_val):
                raise ValueError(f'val ({val}) must be between min_val ({min_val}) and max_val ({max_val})')
            if min_val == max_val:
                return val
        # de-normalizing (check val within limits)
        else:
            if np.any(val < val_lower_limit) or np.any(val > val_upper_limit):
                raise ValueError(f'val ({val}) must be between val_lower_limit ({val_lower_limit}) '
                                 f'and val_upper_limit ({val_upper_limit})')
        return func(val, min_val, max_val)
    return wrapper


@check_vals
def normalize_zero_to_one(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)


@check_vals
def normalize_neg_one_to_one(val, min_val, max_val):
    return normalize_zero_to_one(val, min_val, max_val) * 2 - 1


@check_vals(val_lower_limit=0, val_upper_limit=1)
def denormalize_zero_to_one(val_norm, min_val, max_val):
    return val_norm * (max_val - min_val) + min_val


@check_vals(val_lower_limit=-1, val_upper_limit=1)
def denormalize_neg_one_to_one(val_norm, min_val, max_val):
    return denormalize_zero_to_one((val_norm + 1) / 2, min_val, max_val)
