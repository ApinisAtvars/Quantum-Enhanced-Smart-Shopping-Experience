"""
    Some handy functions for pytroch model training ...
"""
import os
import torch


# Checkpoints
def save_checkpoint(model, model_dir):
    dir = os.path.dirname(model_dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, use_cuda, device_id=None, strict=True, allowed_missing_prefixes=None):
    if use_cuda:
        map_location = None if device_id is None else f'cuda:{device_id}'
    else:
        map_location = 'cpu'

    state_dict = torch.load(model_dir, map_location=map_location)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    if strict:
        model.load_state_dict(state_dict)
        return

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = allowed_missing_prefixes or []
    disallowed_missing = [
        key for key in missing_keys
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]

    if disallowed_missing or unexpected_keys:
        raise RuntimeError(
            'Checkpoint loading failed with non-allowed key mismatches. '
            f'Missing keys: {disallowed_missing}. Unexpected keys: {unexpected_keys}'
        )

    if missing_keys:
        print(f'Loaded checkpoint with allowed missing keys: {missing_keys}')


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                                          lr=params['adam_lr'],
                                                          weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer