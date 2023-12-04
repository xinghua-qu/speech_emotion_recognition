import yaml

class ConfigObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def read_config(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return ConfigObject(config_dict)

from torch.utils.data import random_split

def split_dataset(dataset, config):
    """
    Splits a dataset into training, validation, and test sets.

    Args:
    - dataset (Dataset): The full dataset to be split.
    - config (dict): Configuration dictionary containing split ratios.

    Returns:
    - train_dataset (Dataset): Training subset.
    - val_dataset (Dataset): Validation subset.
    - test_dataset (Dataset): Test subset.
    """

    # Extract split ratios from config
    train_ratio = getattr(config, 'train_ratio', 0.8)
    val_ratio = getattr(config, 'train_ratio', 0.1)
    test_ratio = getattr(config, 'train_ratio', 0.1)
    
    # Calculate sizes of splits
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset
