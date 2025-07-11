import torch
import sys
import os
MLSP_ROOT = os.getenv('MLSP_ROOT')
MLSP_DATA_ROOT = os.getenv('MLSP_DATA_ROOT')
sys.path.append(os.getcwd())


from utils.PAMAP2 import PAMAP
from utils.MHEALTH import MHEALTH
from utils.REALDISP import REALDISP
from utils.discriminator_4_classes import construct_dataset
import numpy as np
import pandas as pd
import torch
import yaml
import h5py
import random
import itertools


def set_random_seed(seed):
    """
    Initialize all the random seeds to a specific number.
    
    Args:
    seed (int): The seed number to initialize the random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Random seed set to {seed}")
def split_list_by_indices(original_list, size_of_part1):
    # Generate a list of indices from 0 to the length of the original list
    indices = list(range(len(original_list)))
    
    # Shuffle the indices to randomize them
    random.shuffle(indices)
    
    # Get indices for part1 and part2
    part1_indices = indices[:size_of_part1]
    part2_indices = indices[size_of_part1:]
    
    # Map these indices back to the original list
    part1 = [original_list[index] for index in part1_indices]
    part2 = [original_list[index] for index in part2_indices]

    return part1, part2

def get_dataset(dataset_name, experiment, current_dir):
    dataset_classes = {'PAMAP2': PAMAP(train = experiment['train'], validation = experiment['validation'] , test = experiment['test'], current_directory = current_dir),
                       'MHEALTH':MHEALTH(train = experiment['train'], validation = experiment['validation'] , test = experiment['test'],PATH = current_dir),
                       'REALDISP':REALDISP(train = experiment['train'], validation = experiment['validation'] , test = experiment['test'],PATH = current_dir),
                       'Opportunity': None}
    
    dataset_class = dataset_classes.get(dataset_name)

    if dataset_class:
        return dataset_class
    else:
        raise ValueError(f"No class defined for {dataset_name}")

import numpy as np
import random

def add_noise(signal, noise_level=0.1):
    """ Add random noise to the signal """
    noise = np.random.randn(*signal.shape) * noise_level
    return signal + noise

def time_shift(signal, shift_max=50):
    """ Randomly shift the signal left or right """
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(signal, shift, axis=1)

def scale_signal(signal, scale_range=(0.9, 1.1)):
    """ Randomly scale the signal """
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return signal * scale_factor

def window_slice(signal, window_size=500):
    """ Randomly slice a window from the signal """
    start = np.random.randint(0, signal.shape[1] - window_size)
    return signal[:, start:start + window_size]

def augment_signal(data):
    """ Apply all augmentations to a dataset of (signal, label1, label2) """
    augmented_data = []
    for signal, label1, label2 in data:
        # Apply augmentations
        print(signal.shape)
        noisy_signal = add_noise(signal)
        shifted_signal = time_shift(noisy_signal)
        scaled_signal = scale_signal(shifted_signal)
        # sliced_signal = window_slice(scaled_signal)

        # Append augmented data
        augmented_data.append((noisy_signal, label1, label2))

    return augmented_data
    
def get_data(dataset_class):
    if dataset_class == None:
        raise ValueError(f"Empty dataset class")
    else:
        dataset_class.get_datasets()
        dataset_class.preprocessing()
        dataset_class.normalize()
        dataset_class.data_segmentation()
        dataset_class.prepare_dataset()

        train = [(a[0],a[1],a[2]) for a in dataset_class.training_final]
        validation = [(a[0],a[1],a[2]) for a in dataset_class.validation_final]
        test = [(a[0],a[1],a[2]) for a in dataset_class.testing_final]

        # data_to_augment = random.sample(train, len(train)//2)

        # augmented_data = augment_signal(data_to_augment)

        # # half_validation, validation = split_list_by_indices(validation, len(validation)//2)

        # for a in augmented_data:
        #     train.append((a[0],a[1],a[2]))
        return {"train": train, "validation": validation, "test": test}
    
def organise_discrimination_data(dataset, samples_per_class):
    return construct_dataset(dataset, samples_per_class)

def save_data(paths, dataset, type, input_parameters):
    if not os.path.exists(paths):
    # Create the directory
        os.makedirs(paths)
        print(f'Directory {paths} created')
    else:
        print(f'Directory {paths} already exists')
    for a in dataset.keys():
        new_path = os.path.join(paths, a)
        print(f"Saving {new_path}")
        # np.save(new_path, dataset[a], allow_pickle=True)
        # Save the list using h5py

        # print(dataset['train'][0][0])
        if type == 0:
            with h5py.File(new_path+'.h5', 'w') as hf:
                for i, (data, activity_label, person_label) in enumerate(dataset[a]):
                    grp = hf.create_group(f'item_{i}')
                    grp.create_dataset('data', data=data)
                    grp.create_dataset('activity_label', data=activity_label)
                    grp.create_dataset('person_label', data=person_label)
        elif type == 1:
            aux = []
            with h5py.File(new_path+f"_{input_parameters['size']}"+'.h5', 'w') as hf:
                for i, (data, disc_label) in enumerate(dataset[a]):
                    aux.append(disc_label)
                    grp = hf.create_group(f'item_{i}')
                    grp.create_dataset('data_0', data=data[0])
                    grp.create_dataset('data_1', data=data[1])
                    grp.create_dataset('disc_label', data=disc_label)
            
            print(np.unique(aux))


def read_distribution(dataset, experiment_number):
    # Read the YAML file
    with open('LOSO_DISTRIBUTIONS.yaml', 'r') as file:
        data = yaml.safe_load(file)

   
    # Get the distribution for the given experiment number
    distribution = data[dataset]['distribution'].get(experiment_number, None)
    
    if distribution is None:
        print(f"No distribution found for experiment number {experiment_number}")
    else:
        print(f"Distribution for experiment {experiment_number}: {distribution}")
    
    return {'train': distribution[0], 'validation': distribution[1], 'test': distribution[2]}

    




if __name__ == "__main__":

    input_parameters = {'dataset': str(sys.argv[1]),
                        'experiment': int(sys.argv[2]),
                        'seed': int(sys.argv[3]),
                        'size': float(sys.argv[4])}
    

    print("\n\n")
    print("------------------------------------------")
    print("We initialize the seed")
    print("------------------------------------------")

    set_random_seed(input_parameters['seed'])
    current_dir = MLSP_DATA_ROOT

    print(f"Data path is {current_dir}")

    data_prepared_dir = os.path.join(current_dir, f'datasets/{input_parameters["dataset"]}/prepared/{input_parameters["experiment"]}')

    paths = {'classification': os.path.join(data_prepared_dir,'classification/') ,
             'discrimination': os.path.join(data_prepared_dir,'discrimination/')}

    # print(f'The path for the classification data is  ̣{paths["classification"]}')
    # print(f'The path for the discrimination data is  ̣{paths["discrimination"]}')

    distribution =read_distribution(input_parameters['dataset'], input_parameters['experiment'])

    print("\n\n")
    print("------------------------------------------")
    print("We construct the classification dataset")
    print("------------------------------------------")

    dataset_object = get_dataset(input_parameters['dataset'], distribution, current_dir)
    classification_dataset = get_data(dataset_object)

    print("\n\n")
    print("------------------------------------------")
    print("We construct the disc dataset")
    print("------------------------------------------")

    discriminator_dataset_train      = construct_dataset(classification_dataset['train'],input_parameters['size'])
    discriminator_dataset_validation = construct_dataset(classification_dataset['validation'],1000)

    labels = []

    for a,b in discriminator_dataset_train:
        labels.append(b)

    print(np.unique(labels))

    discriminator_dataset = {"train": discriminator_dataset_train,
                             "validation": discriminator_dataset_validation}
    
    print("\n\n")
    print("------------------------------------------")
    print("We save the datasets\n")
    print(f'The path for the classification data is  ̣{paths["classification"]}')
    print(f'The path for the discrimination data is  ̣{paths["discrimination"]}')
    print("------------------------------------------")

    save_data(paths["classification"], classification_dataset,0, input_parameters)
    save_data(paths["discrimination"], discriminator_dataset,1,input_parameters)
    


