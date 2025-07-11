import torch
import sys
import os
sys.path.append(os.getcwd())
MLSP_ROOT = os.getenv('MLSP_ROOT')
MLSP_DATA_ROOT = os.getenv('MLSP_DATA_ROOT')
from utils.model.feature_extractor import Feature_Extractor
from utils.model.decoder import Decoder
from utils.model.decoder_realdisp import Decoder_realdisp
import torch.nn as nn
import torch.optim as optim
import h5py
from torch.utils.data import DataLoader
import numpy as np



def get_encoder(channels, seed):
    return Feature_Extractor(input_shape = (64,1,128) , num_blocks = [2,2,0,0], in_channel = channels, seed = seed)

def get_decoder(channels, seed, dataset):
    if dataset == 'REALDISP':
        decoder = Decoder_realdisp(input_shape = (6,1,128), num_blocks=[2,2,0,0],out_channel = channels, seed = seed)
    else:
        decoder = Decoder(input_shape = (6,1,128), num_blocks=[2,2,0,0],out_channel = channels, seed = seed) 
    return decoder

def get_dataset(dataset, experiment, bs, PATH):
    path_train = PATH + f'datasets/{dataset}/prepared/{experiment}/classification/train.h5'
    path_validation = PATH+ f'datasets/{dataset}/prepared/{experiment}/classification/validation.h5'

    dataset_train, dataset_validation, dataset_images = [],[],[]
    with h5py.File(path_train,'r') as file:
        # print("Keys: %s" % list(file.keys()))

        for group in file.keys():
            # print(f"Reading {group}")

            grp = file[group]

            # print(grp['data'])

            data_train = grp['data'][:]
            activity_label_train = grp['activity_label'][()]
            person_label_train = grp['person_label'][()]
            dataset_train.append((data_train, activity_label_train, person_label_train))
    
    with h5py.File(path_validation,'r') as file:
        # print("Keys: %s" % list(file.keys()))

        for group in file.keys():
            # print(f"Reading {group}")

            grp = file[group]

            data_validation = grp['data'][:]
            activity_label_validation = grp['activity_label'][()]
            person_label_validation = grp['person_label'][()]
            dataset_validation.append((data_validation, activity_label_validation, person_label_validation))
            
    
    dataset_images = dataset_validation[0:5]


         

    train_dataloader = DataLoader(dataset_train, 
                                  bs, drop_last=True)
    validation_dataloader = DataLoader(dataset_validation,
                                       64, drop_last=True)
    
    validation_images_dataloader = DataLoader(dataset_images,
                                       1,drop_last=True)
    

    
    return train_dataloader, validation_dataloader, validation_images_dataloader

class Trainer_Step_1():
    def __init__(self, encoder, decoder, train_dataloader, val_dataloader, image_dataloader, device, optimizer, criterion, dataset, experiment, seed, id) -> None:
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        self.val_dataloader = val_dataloader
        self.train_dataloader = train_dataloader
        self.image_dataloader = image_dataloader

        self.device = device
        self.optimizer_reconstruction = optimizer["reconstruction"]
        self.criterion_reconstruction = criterion["reconstruction"]

        self.dataset = dataset
        self.experiment = experiment
        self.seed = seed
        self.id = id
    
    def model(self, encoder, decoder):
        return lambda x: decoder(encoder(x))
    
    def train_one_epoch(self):
        running_loss_reconstruction = 0.0

        samples_per_epoch = 0
        correct_predictions = 0

        avg_loss_reconstruction = 0.0

        self.encoder.train(True)
        self.decoder.train(True)
        
        for i, data in enumerate(self.train_dataloader):
            inputs, activity_class, person = data

            inputs = inputs.to(self.device).float()
            activity_class = activity_class.to(self.device)

            outputs_encoder,_ = self.encoder(inputs)
            outputs_reconstruction = self.decoder(outputs_encoder)


            loss_reconstruction = self.criterion_reconstruction(outputs_reconstruction, torch.unsqueeze(inputs, 2))
            
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            loss_reconstruction.backward()
            self.optimizer_reconstruction.step()

            running_loss_reconstruction += loss_reconstruction.item()
            samples_per_epoch += activity_class.size(0)
        avg_loss_reconstruction = running_loss_reconstruction/samples_per_epoch
        return avg_loss_reconstruction
    
    def validation(self):
        self.encoder.eval()
        self.decoder.eval()
        running_validation_loss_reconstruction = 0.0
        samples = 0
        avg_validation_loss_reconstruction = 0

        with torch.no_grad():
            for i, vdata in enumerate(self.val_dataloader):
                inputs, activity_class, person = vdata
                inputs = inputs.to(self.device).float()
                activity_class = activity_class.to(self.device)

                outputs_encoder,_ = self.encoder(inputs)
                outputs_reconstruction = self.decoder(outputs_encoder)

                validation_loss_reconstruction = self.criterion_reconstruction(outputs_reconstruction, torch.unsqueeze(inputs,2))
                running_validation_loss_reconstruction += validation_loss_reconstruction.item()
                samples += activity_class.size(0)
            avg_validation_loss_reconstruction = running_validation_loss_reconstruction/samples
        return avg_validation_loss_reconstruction
    
    def test_a_few(self):
        self.encoder.eval()
        self.decoder.eval()

        data_original = []
        data_reconstructed = []

        with torch.no_grad():
            for i, vdata in enumerate(self.image_dataloader):
                inputs, activity_class, person = vdata
                inputs = inputs.to(self.device).float()
                activity_class = activity_class.to(self.device)

                outputs_encoder,_ = self.encoder(inputs)
                outputs_reconstruction = self.decoder(outputs_encoder)

                data_original.append(inputs)
                data_reconstructed.append(outputs_reconstruction)

        return data_original, data_reconstructed
    
    def path_existance(self, path):
        if not os.path.exists(path):
            # Create the directory
            os.makedirs(path)
            print(f'Directory {path} created')
        else:
            print(f'Directory {path} already exists')
    
    def train(self, epochs):
        for a in range(epochs):
            avg_loss_reconstruction= self.train_one_epoch()
            avg_loss_reconstruction_val= self.validation()
            print(f"||| Loss {avg_loss_reconstruction} | {avg_loss_reconstruction_val} |||")
            # if a % 3 ==0:
            #     data_original, data_reconstructed = self.test_a_few()
            #     self.plot_signals(a, data_original, data_reconstructed)
        return self.encoder, self.decoder

def save_models(model, path):
    torch.save(model, path)



if __name__ == "__main__":
    input_parameters = {"dataset": str(sys.argv[1]), "experiment": int(sys.argv[2]), "seed": int(sys.argv[3]), "lr": float(sys.argv[4]), 'bs': int(sys.argv[5]), 'device': str(sys.argv[6])}

    sensors_channels = {"PAMAP2": 18, "MHEALTH": 15, "REALDISP":54}

    encoder = get_encoder(sensors_channels[input_parameters["dataset"]], input_parameters['seed'])
    decoder = get_decoder(sensors_channels[input_parameters["dataset"]], input_parameters['seed'])

    criterion = {"reconstruction": nn.MSELoss()}
    
    optimizer = {"reconstruction": optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr= input_parameters['lr'], weight_decay= 10e-5)}

    train_dataloader, validation_dataloader, images_dataloader = get_dataset(input_parameters['dataset'],
                                                          input_parameters['experiment'],
                                                          input_parameters['bs'])
    trainerA = Trainer_Step_1(encoder,decoder, 
                       train_dataloader,
                       validation_dataloader,
                       images_dataloader,
                       input_parameters['device'], 
                       optimizer, 
                       criterion,
                       input_parameters['dataset'],
                       input_parameters['experiment'],
                       input_parameters['seed'])
    encoder, decoder = trainerA.train(40)

    ##Now we should save the models to use them in the second step

    path = MLSP_DATA_ROOT + f'/models/MLSP2025/{input_parameters["dataset"]}/{input_parameters["experiment"]}/{input_parameters["seed"]}/step_1/'

    if not os.path.exists(path):
    # Create the directory
        os.makedirs(path)
        print(f'Directory {path} created')
    else:
        print(f'Directory {path} already exists')


    save_models(encoder, path + 'encoder.pt')
    save_models(decoder, path + 'decoder.pt')


    
