"""Generic helper functions used across codebase."""

import os
import pathlib

import numpy as np
import torch


# Generic.
def get_single_col_by_input_type(input_type, column_definition):
    """Returns name of single column.
    
    Args:
        input_type: Input type of column to extract.
        column_definition: Column definition list for experiment.
    """

    l = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l) != 1:
        raise ValueError('Invalid number of columns for {}'.format(input_type))
    
    return l[0]


def extract_cols_from_data_type(data_type, column_definition,
                                excluded_input_types):
    """Extracts the names of columns that correspond to a define data_type.

    Args:
        data_type: DataType of columns to extract.
        column_definition: Column definition to use.
        excluded_input_types: Set of input types to exclude.

    Returns:
        List of names for columns with data type specified.
    """
    return [
        tup[0]
        for tup in column_definition
        if tup[1] == data_type and tup[2] not in excluded_input_types
    ]


# Loss functions.
def pytorch_quantile_loss(y, y_pred, quantile):
    """Computes quantile loss for pytorch.

    Standard quantile loss as defined in the "Training Procedure" section of
    the main TFT paper

    Args:
        y: Targets
        y_pred: Predictions
        quantile: Quantile to use for loss calculations (between 0 & 1)

    Returns:
        Tensor for quantile loss.
    """

    # Checks quantile
    if quantile < 0 or quantile > 1:
        raise ValueError('Illegal quantile value={}! Values should be between 0 and 1.'
                         .format(quantile))
    
    prediction_underflow = y - y_pred
    q_loss = quantile * torch.relu(prediction_underflow) + (
        1. - quantile) * torch.relu(-prediction_underflow)
    
    return torch.sum(q_loss, dim=-1)


def numpy_normalized_quantile_loss(y, y_pred, quantile):
    """Computes normalized quantile loss for numpy arrays.

    Uses the q-Risk metric as defined in the "Training Procedure" section of the
    main TFT paper.

    Args:
        y: Targets
        y_pred: Predictions
        quantile: Quantile to use for loss calculations (between 0 & 1)

    Returns:
        Float for normalized quantile loss.
    """
    prediction_underflow = y - y_pred
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.) + \
        (1. - quantile) * np.maximum(-prediction_underflow, 0.)
    
    quantile_loss = weighted_errors.mean()
    normalizer = y.abs().mean()

    return 2 * quantile_loss / normalizer


# OS related functions.
def create_folder_if_not_exist(directory):
    """Creates folder if it doesn't exist.

    Args:
        directory: Folder path to create.
    """
    # Also creates directories recursively
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


# Pytorch related functions.
def get_default_pytorch_config(torch_device='gpu', gpu_id=0):
    """Creates pytorch config for graphs to run on CPU or GPU.

    Specifies whether to run graph on gpu or cpu and which GPU ID to use for multi
    GPU machines.

    Args:
        torch_device: 'cpu' or 'gpu'
        gpu_id: GPU ID to use if relevant

    Returns:
        Pytorch config.
    """

    if torch_device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu_id}')
    
    print(f'Selecting {device} (ID={gpu_id})')

    return device


def save(pytorch_model, optimizer, model_folder, cp_name):
    """Saves PyTorch model and optimizer states to checkpoint.

    Args:
        pytorch_model: PyTorch model to be saved.
        optimizer: PyTorch optimizer associated with the model.
        model_folder: Folder to save models.
        cp_name: Name of the checkpoint.
    """
    # Save model and optimizer states
    checkpoint_path = os.path.join(model_folder, f'{cp_name}.pth')
    checkpoint = {'model_state_dict': pytorch_model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, checkpoint_path)
    print(f'Model and optimizer states saved to: {checkpoint_path}')


def load(pytorch_model, optimizer, model_folder, cp_name, verbose=False):
    """Loads PyTorch model and optimizer states from checkpoint.

    Args:
        pytorch_model: PyTorch model to load state into.
        optimizer: PyTorch optimizer to load state into.
        model_folder: Folder containing serialized model.
        cp_name: Name of checkpoint.
        verbose: Whether to print additional debugging information.
    """
    # Load model and optimizer states
    load_path = os.path.join(model_folder, f'{cp_name}.pth')
    checkpoint = torch.load(load_path)
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f'Loaded model and optimizer states from {load_path}')

    if verbose:
        print("Loaded model's state_dict:")
        for param_tensor in pytorch_model.state_dict():
            print(param_tensor, "\t", pytorch_model.state_dict()[param_tensor].size())
        print("Loaded optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])
    
    print('Done.')

    return pytorch_model, optimizer


def print_weights_in_checkpoint(model_folder, cp_name):
    """Prints all weights in PyTorch checkpoint.

    Args:
        model_folder: Folder containing checkpoint
        cp_name: Name of checkpoint

    Returns:
    """
    load_path = os.path.join(model_folder, f'{cp_name}.pth')
    checkpoint = torch.load(load_path)

    print("Weights in PyTorch checkpoint:")
    for name, param in checkpoint['model_state_dict'].items():
        print(f"Parameter name: {name}")
        print(f"Parameter shape: {param.shape}")
        print("-" * 30)