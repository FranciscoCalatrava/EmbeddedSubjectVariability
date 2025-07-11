'''

This code contains the adverarial training following the strategies per batches. That is, first we train D in a batch of all the same person, after in a batch
of all the different person and finally the fooling part.

sorry for typos

'''


import torch
import sys
import os
sys.path.append(os.getcwd())
from utils.model.feature_extractor import Feature_Extractor
from utils.model.decoder import Decoder
from utils.model.classifier import Classifier
from utils.model.discriminator import Discriminator_1
from src.test import Tester
import torch.nn as nn
import torch.optim as optim
import h5py
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.feature_extractor.classifier.save_latent import transform_into_list_classifier, save_latent_space_classifier
from utils.feature_extractor.discriminator.save_latent_discriminator import transform_into_list_discriminator, save_latent_space_discriminator
from itertools import cycle
# import wandb





class Trainer_3():
    def __init__(self, encoder, decoder, discriminator, classifier, train_dataloader_discriminator_positive,train_dataloader_discriminator_negative, val_dataloader_discriminator, train_dataloader_classifier, val_dataloader_classifier, device, optimizer, criterion, weight, dataset, experiment, seed, id) -> None:
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.classifier = classifier.to(device)
        self.discriminator = discriminator.to(device)

        self.train_dataloader_discriminator_positive = train_dataloader_discriminator_positive
        self.train_dataloader_discriminator_negative = train_dataloader_discriminator_negative
        self.val_dataloader_discriminator = val_dataloader_discriminator

        self.train_dataloader_classifier = train_dataloader_classifier
        self.val_dataloader_classifier = val_dataloader_classifier

        self.device = device
        self.experiment = experiment
        self.dataset = dataset
        self.seed = seed
        self.id = id

        self.optimizer_classifier = optimizer["classifier"]
        self.optimizer_encoder = optimizer["encoder"]
        self.optimizer_discriminator = optimizer["discriminator"]

        self.criterion_classifier = criterion["classifier"]
        self.criterion_discriminator = criterion["discriminator"]
        self.criterion_reconstruction = criterion["reconstruction"]

        self.weight_classifier = weight["classifier"]
        self.weight_discriminator = weight["discriminator"]
        self.weight_reconstruction = weight["reconstruction"]

        




    def FADA_g(self, output, targets):
        Discriminator = targets.to(self.device).long()
        labels_1 = torch.where(Discriminator ==  0)

        # print(Discriminator.shape)
        Discriminator[labels_1] = 1
        # print(otuput.shape)
        # print(Discriminator.shape)
        return self.criterion_discriminator(output[labels_1], Discriminator[labels_1])

    def train_one_epoch(self):
        running_loss_3_1 = {"discriminator": 0.0,
        "classifier": 0.0,
        "reconstruction": 0.0,
        "total": 0.0}

        samples_per_epoch_3_1 = {"discriminator": 0,
        "classifier": 0,
        "reconstruction": 0}

        avg_loss_3_1 = {"discriminator": 0.0,
        "classifier": 0.0,
        "reconstruction": 0.0,
        "total": 0.0}

        correct_predictions_3_1 = {"discriminator":0,
        "classifier":0}

        prediction_list_3_1 = {"discriminator": [],
        "classifier": []}

        target_list_3_1 = {"discriminator": [],
        "classifier": []}

        accuracy_3_1 = {"discriminator": 0.0,
        "classifier": 0.0}

        conf_matrix_3_1 = {"discriminator": 0,
        "classifier": 0}


        running_loss_3_2 = {"discriminator": 0.0,
        "classifier": 0.0,
        "reconstruction": 0.0,
        "total": 0.0}

        samples_per_epoch_3_2 = {"discriminator": 0,
        "classifier": 0,
        "reconstruction": 0}

        avg_loss_3_2 = {"discriminator": 0.0,
        "classifier": 0.0,
        "reconstruction": 0.0,
        "total": 0.0}

        correct_predictions_3_2 = {"discriminator":0,
        "classifier":0}

        prediction_list_3_2 = {"discriminator": [],
        "classifier": []}

        target_list_3_2 = {"discriminator": [],
        "classifier": []}

        accuracy_3_2 = {"discriminator": 0.0,
        "classifier": 0.0}

        conf_matrix_3_2 = {"discriminator": 0,
        "classifier": 0}

        discriminator_loss, feature_extractor_loss = [],[]
        softmax = nn.Softmax(dim = 1)
        sigmoid = nn.Sigmoid()


        self.encoder.train(True)
        self.decoder.train(True)
        self.discriminator.train(True)
        self.classifier.train(True)

        dataloader_discriminator_positive = self.train_dataloader_discriminator_positive
        dataloader_discriminator_negative = self.train_dataloader_discriminator_negative
        dataloader_classifier = self.train_dataloader_classifier

        
        for batch_discriminator_positive, batch_discriminator_negative, batch_classifier in zip(dataloader_discriminator_positive, dataloader_discriminator_negative, dataloader_classifier):

        

            ######## We train the discriminator with a batch from same person and same activity
            self.encoder.eval()
            self.classifier.eval()
            self.decoder.eval()
            self.discriminator.train(True)
            for param in self.discriminator.parameters():
                param.requires_grad = True
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

            inputs,targets = batch_discriminator_positive
            inputs[0] = inputs[0].to(self.device).float()
            inputs[1] = inputs[1].to(self.device).float()
            discriminator_labels = targets.to(self.device).long()


            outputs_encoder_0_3_2,_ = self.encoder(inputs[0])
            outputs_encoder_1_3_2,_ = self.encoder(inputs[1])
            inputs_discriminator_3_2 = torch.cat([outputs_encoder_0_3_2, outputs_encoder_1_3_2], dim = 1).to(self.device)

            outputs_discriminator_3_2 = self.discriminator(inputs_discriminator_3_2.detach())
            loss = self.criterion_discriminator(outputs_discriminator_3_2,discriminator_labels)
            loss_same_person = loss
            
            self.discriminator.zero_grad()
            loss.backward()
            self.optimizer_discriminator.step()

            running_loss_3_2["discriminator"] += loss.item()

            predictions_discriminator = torch.argmax(softmax(outputs_discriminator_3_2), dim = 1)
            # (sigmoid(outputs_discriminator_3_2) > 0.5).int()

            correct_predictions_3_2["discriminator"] += (torch.squeeze(predictions_discriminator) == discriminator_labels).sum()

            prediction_list_3_2["discriminator"].append(torch.squeeze(predictions_discriminator))

            target_list_3_2["discriminator"].append(discriminator_labels)

            samples_per_epoch_3_2["discriminator"] += discriminator_labels.size(0)

            ######## We train the discriminator with a batch from DIFFERENT PERSON SAME ACTIVITY

            inputs,targets = batch_discriminator_negative
            inputs[0] = inputs[0].to(self.device).float()
            inputs[1] = inputs[1].to(self.device).float()
            discriminator_labels = targets.to(self.device).long()


            outputs_encoder_0_3_2,_ = self.encoder(inputs[0])
            outputs_encoder_1_3_2,_ = self.encoder(inputs[1])
            inputs_discriminator_3_2 = torch.cat([outputs_encoder_0_3_2, outputs_encoder_1_3_2], dim = 1).to(self.device)

            outputs_discriminator_3_2 = self.discriminator(inputs_discriminator_3_2.detach())
            loss = self.criterion_discriminator(outputs_discriminator_3_2,discriminator_labels)
            loss_different_person = loss
            
            self.discriminator.zero_grad()
            loss.backward()
            self.optimizer_discriminator.step()

            running_loss_3_2["discriminator"] += loss.item()

            predictions_discriminator = torch.argmax(softmax(outputs_discriminator_3_2), dim = 1)
            # (sigmoid(outputs_discriminator_3_2) > 0.5).int()

            correct_predictions_3_2["discriminator"] += (torch.squeeze(predictions_discriminator) == discriminator_labels).sum()

            prediction_list_3_2["discriminator"].append(torch.squeeze(predictions_discriminator))

            target_list_3_2["discriminator"].append(discriminator_labels)

            samples_per_epoch_3_2["discriminator"] += discriminator_labels.size(0)

            ## We update the generator as it should

            self.encoder.train(True)
            self.decoder.eval()
            self.classifier.train(True)
            self.discriminator.eval()
            for param in self.discriminator.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = False

            inputs_classifier, targets_classifier = batch_classifier[0].to(self.device).float(), batch_classifier[1].to(self.device).long() 

            outputs_encoder_0_3_1,_ = self.encoder(inputs[0])
            outputs_encoder_1_3_1,_ = self.encoder(inputs[1])
            outputs_encoder_2_3_1,_ = self.encoder(inputs_classifier)

            outputs_decoder_0_3_1 = self.decoder(outputs_encoder_0_3_1)
            outputs_decoder_1_3_1 = self.decoder(outputs_encoder_1_3_1)

            inputs_discriminator_3_1 = torch.cat([outputs_encoder_0_3_1, outputs_encoder_1_3_1], dim = 1).to(self.device)
            
            outputs_discriminator_3_1 = self.discriminator(inputs_discriminator_3_1)
            outputs_classifier_3_1 = self.classifier(outputs_encoder_2_3_1)

            loss_discriminator_3_1 = self.FADA_g(outputs_discriminator_3_1, targets)
            #self.criterion_discriminator(outputs_discriminator_3_1, discriminator_labels)
            # self.FADA_g(outputs_discriminator_3_1, targets)
            loss_classifier_3_1 = self.criterion_classifier(outputs_classifier_3_1, targets_classifier)
            loss_reconstruction_0 = self.criterion_reconstruction(outputs_decoder_0_3_1, torch.unsqueeze(inputs[0], 2))
            loss_reconstruction_1 = self.criterion_reconstruction(outputs_decoder_1_3_1, torch.unsqueeze(inputs[1], 2))

            loss = self.weight_discriminator*loss_discriminator_3_1 + self.weight_classifier*loss_classifier_3_1 + self.weight_reconstruction*(loss_reconstruction_0 +loss_reconstruction_1)/2
            loss_feature = loss
            self.encoder.zero_grad()
            self.classifier.zero_grad()
            loss.backward()
            self.optimizer_classifier.step()
            self.optimizer_encoder.step()

            running_loss_3_1["discriminator"]+= self.weight_discriminator*loss_discriminator_3_1.item()
            running_loss_3_1["classifier"] += self.weight_classifier*loss_classifier_3_1.item()
            running_loss_3_1["reconstruction"] +=self.weight_reconstruction*(loss_reconstruction_0.item() +loss_reconstruction_1.item())/2
            running_loss_3_1["total"] += loss.item()

            # self.plot_grad_flow(self.encoder.parameters())
            # self.plot_grad_flow(self.classifier.parameters())
            # self.plot_grad_flow(self.DCD.parameters())

            # for name, parameter in self.encoder.named_parameters():
            #     if parameter.requires_grad and parameter.grad is not None:
            #         print(f"{name} - Gradient L2 Norm: {parameter.grad.norm()}")

            predictions_discriminator = torch.argmax(softmax(outputs_discriminator_3_1), dim = 1)
            # (sigmoid(outputs_discriminator_3_1) > 0.5).int()
            predictions_classifier_1 = torch.argmax(softmax(outputs_classifier_3_1), dim = 1)

            correct_predictions_3_1["discriminator"] += (torch.squeeze(predictions_discriminator) == discriminator_labels).sum()
            correct_predictions_3_1["classifier"] += (torch.squeeze(predictions_classifier_1) == targets_classifier).sum()

            prediction_list_3_1["discriminator"].append(torch.squeeze(predictions_discriminator))
            prediction_list_3_1["classifier"].append(torch.squeeze(predictions_classifier_1))


            target_list_3_1["discriminator"].append(discriminator_labels)
            target_list_3_1["classifier"].append(targets_classifier)

            samples_per_epoch_3_1["discriminator"] += discriminator_labels.size(0)
            samples_per_epoch_3_1["classifier"] += targets_classifier.size(0)
            samples_per_epoch_3_1["reconstruction"] += discriminator_labels.size(0) + discriminator_labels.size(0)
            discriminator_loss.append(loss_different_person+loss_same_person)
            feature_extractor_loss.append(loss_feature)
            

        accuracy_3_1["discriminator"] = correct_predictions_3_1["discriminator"].item()/samples_per_epoch_3_1["discriminator"]
        accuracy_3_1["classifier"] = correct_predictions_3_1["classifier"].item()/samples_per_epoch_3_1["classifier"]
        avg_loss_3_1["discriminator"] = running_loss_3_1["discriminator"]/samples_per_epoch_3_1["discriminator"]
        avg_loss_3_1["classifier"] = running_loss_3_1["classifier"]/samples_per_epoch_3_1["classifier"]
        avg_loss_3_1["reconstruction"] = running_loss_3_1["reconstruction"]/(samples_per_epoch_3_1["discriminator"]*2)
        conf_matrix_3_1["discriminator"] = confusion_matrix(torch.cat(target_list_3_1["discriminator"], dim=0).cpu().numpy(), torch.cat(prediction_list_3_1["discriminator"], dim = 0).cpu().numpy())
        conf_matrix_3_1["classifier"] = confusion_matrix(torch.cat(target_list_3_1["classifier"], dim=0).cpu().numpy(), torch.cat(prediction_list_3_1["classifier"], dim = 0).cpu().numpy())

        accuracy_3_2["discriminator"] = correct_predictions_3_2["discriminator"].item()/samples_per_epoch_3_2["discriminator"]
        avg_loss_3_2["discriminator"] = running_loss_3_2["discriminator"]/samples_per_epoch_3_2["discriminator"]
        conf_matrix_3_2["discriminator"] = confusion_matrix(torch.cat(target_list_3_2["discriminator"], dim=0).cpu().numpy(), torch.cat(prediction_list_3_2["discriminator"], dim = 0).cpu().numpy())

        return accuracy_3_1, accuracy_3_2, avg_loss_3_1, avg_loss_3_2,conf_matrix_3_1, conf_matrix_3_2,discriminator_loss, feature_extractor_loss

    def validation_discriminator(self):
        self.discriminator.eval()

        running_loss = {"discriminator": 0.0,
        "reconstruction":0.0}
        samples_per_epoch = {"discriminator": 0,
        "reconstruction":0}
        avg_loss = {"discriminator": 0.0,
        "reconstruction":0.0}
        correct_predictions = {"discriminator":0}
        prediction_list = {"discriminator": []}
        target_list = {"discriminator": []}
        accuracy = {"discriminator": 0.0}
        conf_matrix = {"discriminator": None}

        sigmoid = nn.Sigmoid()
        softmax = nn.Softmax(dim = 1)

        with torch.no_grad():
            for discriminator_batch in self.val_dataloader_discriminator:
                inputs,targets = discriminator_batch
                inputs[0] = inputs[0].to(self.device).float()
                inputs[1] = inputs[1].to(self.device).float()

                discriminator_labels = targets.to(self.device).long()

                ## We get the latent space of both elements of the pair
                outputs_encoder_0,_ = self.encoder(inputs[0])
                outputs_encoder_1,_ = self.encoder(inputs[1])

                ## We reconstruct the signal
                outputs_decoder_0 = self.decoder(outputs_encoder_0)
                outputs_decoder_1 = self.decoder(outputs_encoder_1)

                ## We construct the input of the discriminator and we get the output
                inputs_discriminator = torch.cat([outputs_encoder_0, outputs_encoder_1], dim = 1).to(self.device)
                outputs_discriminator = self.discriminator(inputs_discriminator.detach())

                ##We get the losses from the discriminator and the reconstruction 
                loss_discriminator = self.criterion_discriminator(outputs_discriminator,discriminator_labels)
                loss_reconstruction_1 = self.criterion_reconstruction(outputs_decoder_0, torch.unsqueeze(inputs[0],2))
                loss_reconstruction_2 = self.criterion_reconstruction(outputs_decoder_1, torch.unsqueeze(inputs[1],2))

                ##We store the losses in a dictionary
                running_loss["reconstruction"] += loss_reconstruction_1.item() +loss_reconstruction_2.item()
                running_loss["discriminator"] += loss_discriminator.item()


                predictions_discriminator = torch.argmax(softmax(outputs_discriminator), dim = 1)
                # (sigmoid(outputs_discriminator) > 0.5).int()

                # print(predictions_classifier_1)
                # print(activity_labels_1)

                correct_predictions["discriminator"] += (torch.squeeze(predictions_discriminator) == discriminator_labels.to(self.device)).sum()

                prediction_list["discriminator"].append(torch.squeeze(predictions_discriminator))


                target_list["discriminator"].append(discriminator_labels)

                samples_per_epoch["discriminator"] += discriminator_labels.size(0)
        accuracy["discriminator"] = correct_predictions["discriminator"].item()/samples_per_epoch["discriminator"]
        avg_loss["discriminator"] = running_loss["discriminator"]/samples_per_epoch["discriminator"]
        avg_loss["reconstruction"] = running_loss["reconstruction"]/(samples_per_epoch["discriminator"]*2)
        conf_matrix["discriminator"] = confusion_matrix(torch.cat(target_list["discriminator"], dim=0).cpu().numpy(), torch.cat(prediction_list["discriminator"], dim = 0).cpu().numpy())
        return accuracy, avg_loss, conf_matrix

    def validation_classifier(self):
        self.encoder.eval()
        self.classifier.eval()

        running_loss = {"classifier": 0.0}

        samples_per_epoch = {"classifier": 0}

        avg_loss = {"classifier": 0.0}

        correct_predictions = {"classifier":0}

        prediction_list = {"classifier": []}

        target_list = {"classifier": []}

        accuracy = {"classifier": 0.0}

        conf_matrix = {"classifier": None}

        softmax = nn.Softmax(dim = 1)
        
        with torch.no_grad():
            for classifier_batch in self.val_dataloader_classifier:
                inputs_classifier, targets_classifier = classifier_batch[0].to(self.device).float(), classifier_batch[1].to(self.device)

                ##We get the input and the output from the classification task
                outputs_encoder_2,_ = self.encoder(inputs_classifier)
                outputs_classifier = self.classifier(outputs_encoder_2)

                ##We get the classification loss
                loss_classifier_1 = self.criterion_classifier(outputs_classifier, targets_classifier)

                ##We store it in the dictionary
                running_loss["classifier"] += loss_classifier_1.item()

                ##We get the ground truth and the prediction of the model
                predictions_classifier_1 = torch.argmax(softmax(outputs_classifier), dim = 1)

                ##We store the correct predictions and the labels
                correct_predictions["classifier"] += (torch.squeeze(predictions_classifier_1) == targets_classifier).sum()
                prediction_list["classifier"].append(torch.squeeze(predictions_classifier_1))
                target_list["classifier"].append(targets_classifier)
                samples_per_epoch["classifier"] += targets_classifier.size(0) 
        accuracy["classifier"] = correct_predictions["classifier"].item()/samples_per_epoch["classifier"]
        avg_loss["classifier"] = running_loss["classifier"]/samples_per_epoch["classifier"]
        conf_matrix["classifier"] = confusion_matrix(torch.cat(target_list["classifier"], dim=0).cpu().numpy(), torch.cat(prediction_list["classifier"], dim = 0).cpu().numpy())
        return accuracy, avg_loss, conf_matrix
    
    def get_latent_discriminator(self):
        self.encoder.eval()
        self.decoder.eval()
        self.classifier.eval()
        self.discriminator.eval()

        train_latent_space = []
        validation_latent_space = []

        with torch.no_grad():
            for discriminator_batch in self.train_dataloader_discriminator:
                inputs,targets = discriminator_batch
                inputs[0] = inputs[0].to(self.device).float()
                inputs[1] = inputs[1].to(self.device).float()
                discriminator_labels = targets.to(self.device)
                outputs_encoder_0, out_0= self.encoder(inputs[0])
                outputs_encoder_1, out_1 = self.encoder(inputs[1])
                inputs_discriminator = torch.cat([outputs_encoder_0, outputs_encoder_1], dim = 1).to(self.device)
                train_latent_space = train_latent_space + transform_into_list_discriminator(inputs_discriminator, discriminator_labels)

        with torch.no_grad():
            for discriminator_batch in self.val_dataloader_discriminator:
                inputs,targets = discriminator_batch
                inputs[0] = inputs[0].to(self.device).float()
                inputs[1] = inputs[1].to(self.device).float()
                discriminator_labels = targets.to(self.device)
                outputs_encoder_0, out_0= self.encoder(inputs[0])
                outputs_encoder_1, out_1 = self.encoder(inputs[1])
                inputs_discriminator = torch.cat([outputs_encoder_0, outputs_encoder_1], dim = 1).to(self.device)
                validation_latent_space = validation_latent_space + transform_into_list_discriminator(inputs_discriminator, discriminator_labels)

        # print(train_latent_space[0:4])

        return train_latent_space, validation_latent_space
    
    def get_latent_classifier(self):
        self.encoder.eval()
        self.classifier.eval()
        validation_latent_space = []

        with torch.no_grad():
            for classifier_batch in self.val_dataloader_classifier:
                inputs_classifier, targets_classifier, participant = classifier_batch[0].to(self.device).float(), classifier_batch[1].to(self.device), classifier_batch[2].to(self.device)
                outputs_encoder_classifier,_ = self.encoder(inputs_classifier)
                validation_latent_space = validation_latent_space + transform_into_list_classifier(outputs_encoder_classifier, targets_classifier, participant)

        return validation_latent_space



    def train(self, epochs, dataset_type):
        # wandb.init(project='Percom', name="Test")
        threshold = 0.5
        threshold_dict = {"PAMAP2": 0.87, "MHEALTH":0.96, "REALDISP":0.95}
        threshold = threshold_dict[dataset_type]


        print(threshold)
        buffer_values = []
        shoud_break = False
        max_value = 0.0
        index_list = []
        scheduler_discriminator = optim.lr_scheduler.StepLR(self.optimizer_discriminator, step_size =5 , gamma = 0.4, verbose = True)
        scheduler_encoder = optim.lr_scheduler.StepLR(self.optimizer_encoder, step_size = 10, gamma = 0.4, verbose = True)
        scheduler_decoder = optim.lr_scheduler.StepLR(self.optimizer_classifier, step_size = 10 , gamma = 0.4, verbose = True)
        for a in range(epochs):
            accuracy_3_1, accuracy_3_2, avg_loss_3_1, avg_loss_3_2,conf_matrix_3_1, conf_matrix_3_2, discriminator_loss, feature_extractor_loss= self.train_one_epoch()
            # wandb.log({"Discriminator_loss": discriminator_loss, "feature_extractor_loss": feature_extractor_loss})
            print(f"|Training| 3.1 | Acc {accuracy_3_1} | Loss {avg_loss_3_1}")
            print(f"|Training| 3.2 | Acc {accuracy_3_2} | Loss {avg_loss_3_2}")


            # if a%5 == 0:
            #     train_latent, validation_latent =self.get_latent_discriminator()
            #     save_latent_space_discriminator(dataset = self.dataset, 
            #                       experiment = self.experiment,
            #                     seed = self.seed, data = train_latent, epoch = a, step = 3, model = "encoder",name = "train")
            #     save_latent_space_discriminator(dataset = self.dataset, 
            #                       experiment = self.experiment,
            #                     seed = self.seed, data = validation_latent, epoch = a, step = 3, model = "encoder",name = "validation")
        
            #     val_classifier = self.get_latent_classifier()
            #     save_latent_space_classifier(dataset = self.dataset, 
            #                             experiment = self.experiment,
            #                             seed = self.seed, data = val_classifier, epoch = a, step = 3, model = "classification",name = "validation")

            if a%1 == 0:

                accuracy_classifier, avg_loss_classifier, conf_matrix_classifier = self.validation_classifier()
                accuracy_discriminator, avg_loss_discriminator, conf_matrix_discriminator = self.validation_discriminator()

                print(f"|Validation| Loss Classifier {avg_loss_classifier['classifier']} Loss Discriminator: {avg_loss_discriminator['discriminator']} loss Reconstruction {avg_loss_discriminator['reconstruction']} ")
                print(f"|Validation| Acc Classifier {accuracy_classifier['classifier']} Acc Discriminator: {accuracy_discriminator['discriminator']}")
                print(conf_matrix_discriminator["discriminator"])



                # wandb.log({"Loss Training Discriminator Step 3.1": avg_loss_3_1['discriminator'], "Loss Training Discriminator Step 3.2": avg_loss_3_2['discriminator'], "Loss Validation Discriminator": avg_loss_discriminator['discriminator']})
                # wandb.log({"Loss Training Classifier Step 3.1": avg_loss_3_1['classifier'], "Loss Training Classifier Step 3.2": avg_loss_3_2['classifier'], "Loss Validation Classifier": avg_loss_classifier['classifier']})
                # wandb.log({"Acc Training Discriminator Step 3.1": accuracy_3_1['discriminator'], "Acc Training Discriminator Step 3.2": accuracy_3_2['discriminator'], "Acc Validation Discriminator": accuracy_discriminator['discriminator']})
                # wandb.log({"Acc Training Classifier Step 3.1": accuracy_3_1['classifier'], "Acc Training Classifier Step 3.2": accuracy_3_2['classifier'], "Acc Validation Classifier": accuracy_classifier['classifier']})

                # testerA = Tester(self.encoder, self.classifier, device= self.device)
                # acc_test = testerA.test(dataloaders[1])
                print(f"The acc_val is{accuracy_classifier['classifier']}")
                buffer_values.append(accuracy_classifier['classifier'])
                print(len(buffer_values))
                if(accuracy_classifier['classifier'] >  threshold and len(buffer_values) > 40):
                    shoud_break = True
                    break
            if shoud_break == True:
                break


            scheduler_discriminator.step()
            scheduler_encoder.step()
            scheduler_decoder.step()
        # wandb.finish()
        return self.encoder, self.decoder, self.discriminator, self.classifier
    
    


def load_model(path):
    return torch.load(path)

def test_path(path):
    if not os.path.exists(path):
        print(f'Directory {path} does not exist')
    else:
        print(f'Directory {path} already exists')

def get_classification_dataset(path):
    dataset_train, dataset_validation, dataset_test = [],[],[]
    with h5py.File(path+"train.h5",'r') as file:
        # print("Keys: %s" % list(file.keys()))

        for group in file.keys():
            # print(f"Reading {group}")

            grp = file[group]

            # print(grp['data'])

            data_train = grp['data'][:]
            activity_label_train = grp['activity_label'][()]
            person_label_train = grp['person_label'][()]
            dataset_train.append((data_train, activity_label_train, person_label_train))
    with h5py.File(path + "validation.h5",'r') as file:
        # print("Keys: %s" % list(file.keys()))

        for group in file.keys():
            # print(f"Reading {group}")

            grp = file[group]

            data_validation = grp['data'][:]
            activity_label_validation = grp['activity_label'][()]
            person_label_validation = grp['person_label'][()]
            dataset_validation.append((data_validation, activity_label_validation, person_label_validation))
    with h5py.File(path + "test.h5",'r') as file:
        print("Keys: %s" % list(file.keys()))

        for group in file.keys():
            # print(f"Reading {group}")

            grp = file[group]

            data_test = grp['data'][:]
            activity_label_test = grp['activity_label'][()]
            person_label_test = grp['person_label'][()]
            dataset_test.append((data_test, activity_label_test, person_label_test))
    return dataset_train, dataset_validation, dataset_test


def get_discriminator_dataset(path):
    dataset_train_positive,dataset_train_negative, dataset_validation = [],[],[]
    with h5py.File(path+"train_positive.h5",'r') as file:
        # print("Keys: %s" % list(file.keys()))

        for group in file.keys():
            # print(f"Reading {group}")

            grp = file[group]

            # print(grp['data'])

            data_0_train = grp['data_0'][()]
            data_1_train = grp['data_1'][()]
            disc_label_train = grp['disc_label'][()]
            dataset_train_positive.append(((data_0_train, data_1_train), disc_label_train))
    with h5py.File(path+"train_negative.h5",'r') as file:
        # print("Keys: %s" % list(file.keys()))

        for group in file.keys():
            # print(f"Reading {group}")

            grp = file[group]

            # print(grp['data'])

            data_0_train = grp['data_0'][()]
            data_1_train = grp['data_1'][()]
            disc_label_train = grp['disc_label'][()]
            dataset_train_negative.append(((data_0_train, data_1_train), disc_label_train))
    with h5py.File(path + "validation.h5",'r') as file:
        # print("Keys: %s" % list(file.keys()))

        for group in file.keys():
            # print(f"Reading {group}")S
            grp = file[group]

            data_0_validation = grp['data_0'][()]
            data_1_validation = grp['data_1'][()]
            disc_label_validation = grp['disc_label'][()]
            dataset_validation.append(((data_0_validation, data_1_validation), disc_label_validation))
    return dataset_train_positive,dataset_train_negative, dataset_validation


def get_dataset_3(dataset, experiment, bs):
    path_classification = f'/datasets/{dataset}/prepared/{experiment}/classification/'
    path_discrimination = f'/datasets/{dataset}/prepared/{experiment}/discrimination/'

    train_classifiaction, validation_classification, test_classification = get_classification_dataset(path_classification)
    train_discrimination_positive,train_discrimination_negative, validation_discrimination = get_discriminator_dataset(path_discrimination)

    final_train_discrimination_positive,final_train_discrimination_negative, final_validation_discrimination = [],[],[]
    label_aux = []
    for pairs, label  in train_discrimination_positive:
        final_train_discrimination_positive.append(((train_classifiaction[pairs[0]][0],train_classifiaction[pairs[1]][0]), label))
        label_aux.append(label)
    print(np.unique(label_aux))

    label_aux = []
    for pairs, label  in train_discrimination_negative:
        final_train_discrimination_negative.append(((train_classifiaction[pairs[0]][0],train_classifiaction[pairs[1]][0]), label))
        label_aux.append(label)
    print(np.unique(label_aux))

    label_aux = []
    for pairs, label  in validation_discrimination:
        final_validation_discrimination.append(((validation_classification[pairs[0]][0],validation_classification[pairs[1]][0]), label))
        label_aux.append(label)
    print(np.unique(label_aux))

    train_dataloader_classification = DataLoader(train_classifiaction, 
                                  bs,shuffle=True)
    validation_dataloader_classification = DataLoader(validation_classification,
                                       64)
    test_dataloader_classification = DataLoader(test_classification,
                                       64)
    
    number_batches_classification = len(train_dataloader_classification)
    bs_discriminator = len(train_discrimination_positive) // number_batches_classification

    train_dataloader_discriminator_positive =  DataLoader(final_train_discrimination_positive, 
                                  bs_discriminator, shuffle=True)

    train_dataloader_discriminator_negative =  DataLoader(final_train_discrimination_negative, 
                                  bs_discriminator, shuffle=True)
    validation_dataloader_discriminator = DataLoader(final_validation_discrimination, 
                                  128)    

    
    return train_dataloader_classification, validation_dataloader_classification, test_dataloader_classification, train_dataloader_discriminator_positive, train_dataloader_discriminator_negative, validation_dataloader_discriminator





def save_models(model, path):
    torch.save(model, path)





if __name__ == "__main__":

    input_parameters = {"dataset": str(sys.argv[1]), 
                        "experiment": int(sys.argv[2]), 
                        "seed": int(sys.argv[3]), 
                        'device': str(sys.argv[4]),
                        'decoder_lr': float(sys.argv[5]),
                        'decoder_bs':int(sys.argv[6]),
                        'classifier_lr': float(sys.argv[7]),
                        'classifier_bs': int(sys.argv[8]),
                        'discriminator_lr': float(sys.argv[9])
                            }
    
    sensors_channels = {"PAMAP2": 18, "MHEALTH": 15, "REALDISP":54}

    

    path_encoder = f"/models/Percom2024/{input_parameters['dataset']}/{input_parameters['experiment']}/{input_parameters['seed']}/step_2/encoder.pt"
    path_decoder = f"/models/Percom2024/{input_parameters['dataset']}/{input_parameters['experiment']}/{input_parameters['seed']}/step_2/decoder.pt"
    path_discriminator = f"/models/Percom2024/{input_parameters['dataset']}/{input_parameters['experiment']}/{input_parameters['seed']}/step_2/discriminator.pt"
    path_classifier = f"/models/Percom2024/{input_parameters['dataset']}/{input_parameters['experiment']}/{input_parameters['seed']}/step_2/classifier.pt"


    ##Load models    
    encoder = load_model(path_encoder)
    decoder = load_model(path_decoder)
    classifier = load_model(path_classifier)
    discriminator = load_model(path_discriminator)

    criterion = {"discriminator": nn.CrossEntropyLoss(),
    "classifier": nn.CrossEntropyLoss(),
    "reconstruction": nn.MSELoss()}

    optimizer = {"discriminator": optim.Adam(discriminator.parameters(), lr = 1e-3),
    "encoder": optim.Adam(encoder.parameters(), lr = 1e-3),
    "classifier": optim.Adam(classifier.parameters(), lr = 1e-3)}

    classification_train_dataloader, classification_validation_dataloader, classification_test_dataloader, discriminator_train_dataloader_positive,discriminator_train_dataloader_negative, discriminator_validation_dataloader = get_dataset(input_parameters['dataset'],
                                                                                                                                                                                             input_parameters['experiment'],
                                                                                                                                                                                             input_parameters['classifier_bs'])
    


    weight = {"reconstruction": 0.7,
    "classifier": 0.2,
    "discriminator": 0.1}
    
    train_3 = Trainer_3(encoder, decoder, discriminator, classifier, discriminator_train_dataloader_positive,
                              discriminator_train_dataloader_negative, 
                              discriminator_validation_dataloader, 
                              classification_train_dataloader,
                              classification_validation_dataloader, 
                              input_parameters['device'], optimizer, criterion, weight,input_parameters['dataset'],
                              input_parameters['experiment'],
                              input_parameters['seed'])
    

    encoder, decoder, discriminator, classifier = train_3.train(75, input_parameters['dataset'])

    ##We get the results for the testset
    testerA = Tester(encoder, classifier, device= input_parameters['device'])
    acc_test = testerA.test(classification_test_dataloader, input_parameters['dataset'], input_parameters['experiment'])
    print(f"The acc_test is{acc_test}")


    path = f'/models/Percom2024/{input_parameters["dataset"]}/{input_parameters["experiment"]}/{input_parameters["seed"]}/step_3/'

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
