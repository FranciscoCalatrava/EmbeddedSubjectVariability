import torch
import sys
import os
MLSP_ROOT = os.getenv('MLSP_ROOT')
MLSP_DATA_ROOT = os.getenv('MLSP_DATA_ROOT')
sys.path.append(os.getcwd())

import torch.nn as nn
import torch.optim as optim

from src.train_step_1 import get_encoder, get_decoder, get_dataset, Trainer_Step_1, save_models
from src.train_step_2_quatity_study import load_model, get_dataset_2,Trainer_Step_2
from src.train_step_3_quantity_study import Trainer_3, get_dataset_3
from src.test import Tester

from utils.model.classifier import Classifier
from utils.model.discriminator import Discriminator_1
import random
import numpy as np


import json

def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



PATH = MLSP_DATA_ROOT




def run_step_1(input_parameters):
    sensors_channels = {"PAMAP2": 18, "MHEALTH": 15, "REALDISP":54}

    encoder = get_encoder(sensors_channels[input_parameters["dataset"]], input_parameters['seed'])
    decoder = get_decoder(sensors_channels[input_parameters["dataset"]], input_parameters['seed'], input_parameters["dataset"])

    criterion = {"reconstruction": nn.MSELoss()}
    
    optimizer = {"reconstruction": optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr= input_parameters['step_1_lr'], weight_decay= 10e-5)}

    train_dataloader, validation_dataloader, images_dataloader = get_dataset(input_parameters['dataset'],
                                                          input_parameters['experiment'],
                                                          input_parameters['step_1_bs'],
                                                          PATH)
    trainerA = Trainer_Step_1(encoder,decoder, 
                       train_dataloader,
                       validation_dataloader,
                       images_dataloader,
                       input_parameters['device'], 
                       optimizer, 
                       criterion,
                       input_parameters['dataset'],
                       input_parameters['experiment'],
                       input_parameters['seed'],
                       input_parameters['id'])
    encoder, decoder = trainerA.train(20)

    ##Now we should save the models to use them in the second step

    path = PATH + f'models/MLSP2025/{input_parameters["dataset"]}/loss_function_study/{input_parameters["experiment"]}/{input_parameters["seed"]}/{input_parameters["id"]}/step_1/'

    if not os.path.exists(path):
    # Create the directory
        os.makedirs(path)
        print(f'Directory {path} created')
    else:
        print(f'Directory {path} already exists')


    save_models(encoder, path + 'encoder.pt')
    save_models(decoder, path + 'decoder.pt')

def run_step_2(input_parameters):
    sensors_channels = {"PAMAP2": 18, "MHEALTH": 15, "REALDISP":54}

    

    path_encoder = PATH + f'models/MLSP2025/{input_parameters["dataset"]}/loss_function_study/{input_parameters["experiment"]}/{input_parameters["seed"]}/{input_parameters["id"]}/step_1/encoder.pt'
    path_decoder = PATH + f'models/MLSP2025/{input_parameters["dataset"]}/loss_function_study/{input_parameters["experiment"]}/{input_parameters["seed"]}/{input_parameters["id"]}/step_1/decoder.pt'


    ##Load models    
    encoder = load_model(path_encoder)
    decoder = load_model(path_decoder)
    classifier = Classifier(sensors_channels[input_parameters['dataset']], input_parameters['seed'])
    discriminator = Discriminator_1(input_parameters['seed'])

    ##We get the dataloaders

    classification_train_dataloader, classification_validation_dataloader, classification_test_dataloader, discriminator_train_dataloader, discriminator_validation_dataloader = get_dataset_2(input_parameters['dataset'],
                                                                                                                                                                                             input_parameters['experiment'],
                                                                                                                                                                                             input_parameters['step_2_classifier_bs'],PATH, input_parameters['trainsize'])



    ## We define the loss functions

    criterion = {"reconstruction": nn.MSELoss(),
    "discriminator": nn.CrossEntropyLoss(),
    "classifier": nn.CrossEntropyLoss()}

    ## We define the optimizers
    
    optimizer = {"encoder": optim.Adam(encoder.parameters() , lr= input_parameters['step_2_decoder_lr'], weight_decay= 10e-5),
    "decoder": optim.Adam(decoder.parameters(), lr= input_parameters['step_2_decoder_lr'], weight_decay= 10e-5),
    "discriminator": optim.Adam(discriminator.parameters(), lr = input_parameters['step_2_discriminator_lr']),
    "classifier": optim.Adam(classifier.parameters(), lr= input_parameters['step_2_classifier_lr'], weight_decay= 10e-5)}

    trainerB = Trainer_Step_2(encoder, 
                              decoder, 
                              classifier, 
                              discriminator, 
                              discriminator_train_dataloader, 
                              discriminator_validation_dataloader, 
                              classification_train_dataloader,
                              classification_validation_dataloader, 
                              input_parameters['device'], 
                              optimizer, 
                              criterion,
                              input_parameters['experiment'],
                              input_parameters['seed'],
                              input_parameters['dataset'],
                              input_parameters['id'] )
    
    encoder, decoder, discriminator, classifier = trainerB.train(10)

    path = PATH + f'models/MLSP2025/{input_parameters["dataset"]}/loss_function_study/{input_parameters["experiment"]}/{input_parameters["seed"]}/{input_parameters["id"]}/step_2/'

    if not os.path.exists(path):
    # Create the directory
        os.makedirs(path)
        print(f'Directory {path} created')
    else:
        print(f'Directory {path} already exists')


    save_models(encoder, path + 'encoder.pt')
    save_models(decoder, path + 'decoder.pt')
    save_models(discriminator, path + 'discriminator.pt')
    save_models(classifier, path + 'classifier.pt')



def run_step_3(input_parameters):
    sensors_channels = {"PAMAP2": 18, "MHEALTH": 15, "REALDISP":54}

    

    path_encoder = PATH + f'models/MLSP2025/{input_parameters["dataset"]}/loss_function_study/{input_parameters["experiment"]}/{input_parameters["seed"]}/{input_parameters["id"]}/step_2/encoder.pt'
    path_decoder = PATH + f'models/MLSP2025/{input_parameters["dataset"]}/loss_function_study/{input_parameters["experiment"]}/{input_parameters["seed"]}/{input_parameters["id"]}/step_2/decoder.pt'
    path_discriminator = PATH + f'models/MLSP2025/{input_parameters["dataset"]}/loss_function_study/{input_parameters["experiment"]}/{input_parameters["seed"]}/{input_parameters["id"]}/step_2/discriminator.pt'
    path_classifier = PATH+ f'models/MLSP2025/{input_parameters["dataset"]}/loss_function_study/{input_parameters["experiment"]}/{input_parameters["seed"]}/{input_parameters["id"]}/step_2/classifier.pt'


    ##Load models    
    encoder = load_model(path_encoder)
    decoder = load_model(path_decoder)
    classifier = load_model(path_classifier)
    discriminator = load_model(path_discriminator)

    criterion = {"discriminator": nn.CrossEntropyLoss(),
    "classifier": nn.CrossEntropyLoss(),
    "reconstruction": nn.MSELoss()}

    optimizer = {"discriminator": optim.Adam(discriminator.parameters(), lr = 1e-4),
    "encoder": optim.Adam(encoder.parameters(), lr = 1e-3),
    "classifier": optim.Adam(classifier.parameters(), lr = 1e-4)}

    classification_train_dataloader, classification_validation_dataloader, classification_test_dataloader, discriminator_train_dataloader, discriminator_validation_dataloader = get_dataset_3(input_parameters['dataset'],
                                                                                                                                                                                             input_parameters['experiment'],
                                                                                                                                                                                             input_parameters['step_2_classifier_bs'],PATH, input_parameters['trainsize'])
    

    weight = {"reconstruction": input_parameters['step_3_reconstruction_weight'],
    "classifier": input_parameters['step_3_classification_weight'],
    "discriminator": input_parameters['step_3_discriminator_weight']}
    
    train_3 = Trainer_3(encoder, decoder, discriminator, classifier, discriminator_train_dataloader, 
                              discriminator_validation_dataloader, 
                              classification_train_dataloader,
                              classification_validation_dataloader,
                              classification_test_dataloader,
                              input_parameters['device'], optimizer, criterion, weight,input_parameters['dataset'],
                              input_parameters['experiment'],
                              input_parameters['seed'],
                              input_parameters['id'])
    

    encoder, decoder, discriminator, classifier = train_3.train(200, input_parameters['dataset'])

    ##We get the results for the testset
    testerA = Tester(encoder, classifier, device= input_parameters['device'])
    acc_test = testerA.test(classification_test_dataloader, input_parameters['dataset'], input_parameters['experiment'])
    print(f"The acc_test is{acc_test}")


    path = PATH + f'models/MLSP2025/{input_parameters["dataset"]}/loss_function_study/{input_parameters["experiment"]}/{input_parameters["seed"]}/{input_parameters["id"]}/step_3/'

    if not os.path.exists(path):
    # Create the directory
        os.makedirs(path)
        print(f'Directory {path} created')
    else:
        print(f'Directory {path} already exists')


    file_path = os.path.join(path, 'hyperparameters.json')

    # Write the dictionary to a file
    with open(file_path, 'w') as file:
        json.dump(input_parameters, file, indent=4)


    save_models(encoder, path + 'encoder.pt')
    save_models(decoder, path + 'decoder.pt')
    save_models(discriminator, path + 'discriminator.pt')
    save_models(classifier, path + 'classifier.pt')






if __name__ == "__main__":
    input_parameters = {"dataset": str(sys.argv[1]), 
    "experiment": int(sys.argv[2]), 
    "seed": int(sys.argv[3]),
    "device": str(sys.argv[4]),
    "step_1_lr": float(sys.argv[5]), 
    'step_1_bs': int(sys.argv[6]),
    'step_2_decoder_lr': float(sys.argv[7]),
    'step_2_decoder_bs':int(sys.argv[8]),
    'step_2_classifier_lr': float(sys.argv[9]),
    'step_2_classifier_bs': int(sys.argv[10]),
    'step_2_discriminator_lr': float(sys.argv[11]),
    'step_3_discriminator_weight': float(sys.argv[12]),
    'step_3_reconstruction_weight': float(sys.argv[13]),
    'step_3_classification_weight': float(sys.argv[14]),
    'id': int(sys.argv[15]),
    'trainsize': float(sys.argv[16])

    }
    seed_everything(input_parameters['seed'])
    run_step_1(input_parameters)
    run_step_2(input_parameters)
    run_step_3(input_parameters)