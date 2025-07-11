import torch
import sys
import os
sys.path.append(os.getcwd())
from utils.model.feature_extractor import Feature_Extractor
from utils.model.decoder import Decoder
from utils.model.classifier import Classifier
from utils.model.discriminator import Discriminator_1
import torch.nn as nn
import torch.optim as optim
import h5py
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.feature_extractor.classifier.save_latent import transform_into_list_classifier, save_latent_space_classifier
from utils.feature_extractor.discriminator.save_latent_discriminator import transform_into_list_discriminator, save_latent_space_discriminator

def euclidean_distance(latent1, latent2):
    return (latent1 - latent2).pow(2).sum(dim=1, keepdim=True).sqrt()

def cosine_similarity(latent1, latent2):
    return (latent1 * latent2).sum(dim=1, keepdim=True) / (
        latent1.norm(p=2, dim=1, keepdim=True) * latent2.norm(p=2, dim=1, keepdim=True))



class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation accuracy improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation accuracy improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = float('-inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        """Saves model when validation accuracy increase."""
        if self.verbose:
            self.trace_func(f'Validation accuracy increased ({self.val_acc_max:.6f}% --> {val_acc:.6f}%). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_acc_max = val_acc



class Trainer_Step_2():
    def __init__(self, encoder, decoder, classifier, discriminator, train_dataloader, val_dataloader, train_dataloader_classifier, val_dataloader_classifier, device, optimizer, criterion, experiment, seed, dataset, id) -> None:
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.classifier = classifier.to(device)
        self.discriminator = discriminator.to(device)

        self.train_dataloader_discriminator = train_dataloader
        self.val_dataloader_discriminator = val_dataloader
        self.train_dataloader_classifier = train_dataloader_classifier
        self.val_dataloader_classifier = val_dataloader_classifier

        self.device = device
        self.experiment = experiment
        self.seed = seed
        self.dataset = dataset
        self.id = id

        self.optimizer_encoder = optimizer["encoder"]
        self.optimizer_decoder = optimizer["decoder"]
        self.optimizer_classifier = optimizer["classifier"]
        self.optimizer_discriminator = optimizer["discriminator"]

        self.criterion_reconstruction = criterion["reconstruction"]
        self.criterion_classifier = criterion["classifier"]
        self.criterion_discriminator = criterion["discriminator"]

    
    def train_one_epoch(self):
        running_loss = {"discriminator": 0.0,
        "classifier": 0.0,
        "reconstruction": 0.0}

        samples_per_epoch = {"discriminator": 0,
        "classifier": 0,
        "reconstruction": 0}

        avg_loss = {"discriminator": 0.0,
        "classifier": 0.0,
        "reconstruction": 0.0}

        correct_predictions = {"discriminator":0,
        "classifier":0}

        prediction_list = {"discriminator": [],
        "classifier": []}

        target_list = {"discriminator": [],
        "classifier": []}

        accuracy = {"discriminator": 0.0,
        "classifier": 0.0}

        conf_matrix = {"discriminator": 0,
        "classifier": 0}

        softmax = nn.Softmax(dim = 1)
        sigmoid = nn.Sigmoid()


        self.encoder.train(True)
        self.decoder.train(True)
        self.discriminator.train(True)
        self.classifier.train(True)

        for discriminator_batch, classifier_batch in zip(self.train_dataloader_discriminator, self.train_dataloader_classifier):
            inputs,targets = discriminator_batch
            inputs[0] = inputs[0].to(self.device).float()
            inputs[1] = inputs[1].to(self.device).float()

            discriminator_labels = targets.to(self.device)

            # print(torch.unique(discriminator_labels))

            inputs_classifier, targets_classifier = classifier_batch[0].to(self.device).float(), classifier_batch[1].to(self.device)

            outputs_encoder_0,_ = self.encoder(inputs[0])
            outputs_encoder_1,_ = self.encoder(inputs[1])
            outputs_encoder_2,_ = self.encoder(inputs_classifier)
            outputs_decoder_0 = self.decoder(outputs_encoder_0)
            outputs_decoder_1 = self.decoder(outputs_encoder_1)
            # print(outputs_encoder_2.shape)
            outputs_classifier = self.classifier(outputs_encoder_2)




            loss_reconstruction_1 = self.criterion_reconstruction(outputs_decoder_0, torch.unsqueeze(inputs[0],2))
            loss_reconstruction_2 = self.criterion_reconstruction(outputs_decoder_1, torch.unsqueeze(inputs[1],2))
            loss_classifier_1 = self.criterion_classifier(outputs_classifier, targets_classifier)

            self.decoder.zero_grad()
            self.encoder.zero_grad()
            self.classifier.zero_grad()
            loss_reconstruction = (loss_reconstruction_1 +loss_reconstruction_2)
            loss_classifier = (loss_classifier_1)
            loss_reconstruction.backward(retain_graph=True)
            loss_classifier.backward()
            self.optimizer_classifier.step()
            self.optimizer_decoder.step()
            self.optimizer_encoder.step()

            running_loss["classifier"] += loss_classifier.item()
            running_loss["reconstruction"] += loss_reconstruction.item()

            predictions_classifier_1 = torch.argmax(softmax(outputs_classifier), dim = 1)

            correct_predictions["classifier"] += (torch.squeeze(predictions_classifier_1) == targets_classifier).sum()

            prediction_list["classifier"].append(torch.squeeze(predictions_classifier_1))


            target_list["classifier"].append(targets_classifier)

            samples_per_epoch["classifier"] += targets_classifier.size(0)
            samples_per_epoch["reconstruction"] +=discriminator_labels.size(0)*2
        accuracy["classifier"] = correct_predictions["classifier"].item()/samples_per_epoch["classifier"]
        avg_loss["classifier"] = running_loss["classifier"]/samples_per_epoch["classifier"]
        avg_loss["reconstruction"] = running_loss["reconstruction"]/samples_per_epoch["reconstruction"]
        # print("Just finished step 2")
        conf_matrix["classifier"] = confusion_matrix(torch.cat(target_list["classifier"], dim=0).cpu().numpy(), torch.cat(prediction_list["classifier"], dim = 0).cpu().numpy())
        # print(f"The avg loss is: {avg_loss} ||| The accuracy from discriminator is: {accuracy_discriminator} ||| The accuracy from classifier is: {accuracy_classifier}")
        return accuracy, avg_loss, conf_matrix
    
    def validation_classifier(self):

        self.encoder.eval()
        self.decoder.eval()
        self.classifier.eval()
        self.discriminator.eval()

        running_loss = {"classifier": 0.0,
        "reconstruction": 0.0}

        samples_per_epoch = {"classifier": 0}

        avg_loss = {"classifier": 0.0,
        "reconstruction": 0.0}

        correct_predictions = {"classifier":0}

        prediction_list = {"classifier": []}

        target_list = {"classifier": []}

        accuracy = {"classifier": 0.0}

        conf_matrix = {"classifier": None}

        softmax = nn.Softmax(dim = 1)
        sigmoid = nn.Sigmoid()

        with torch.no_grad():
            for classifier_batch in self.val_dataloader_classifier:

                inputs_classifier, targets_classifier = classifier_batch[0].to(self.device).float(), classifier_batch[1].to(self.device)

                # print(targets_classifier)

                outputs_encoder_classifier,_ = self.encoder(inputs_classifier)
                output_decoder_classifier= self.decoder(outputs_encoder_classifier)
                outputs_classifier = self.classifier(outputs_encoder_classifier)

                loss_reconstruction = self.criterion_reconstruction(output_decoder_classifier, torch.unsqueeze(inputs_classifier, 2))
                loss_classifier_1 = self.criterion_classifier(outputs_classifier, targets_classifier)

                running_loss["classifier"] += loss_classifier_1.item()
                running_loss["reconstruction"] += loss_reconstruction.item()

                predictions_classifier_1 = torch.argmax(softmax(outputs_classifier), dim = 1)

                # print(predictions_classifier_1)
                # print(activity_labels_1)

                correct_predictions["classifier"] += (torch.squeeze(predictions_classifier_1) == targets_classifier).sum()

                prediction_list["classifier"].append(torch.squeeze(predictions_classifier_1))


                target_list["classifier"].append(targets_classifier)

                samples_per_epoch["classifier"] += targets_classifier.size(0)
        accuracy["classifier"] = correct_predictions["classifier"].item()/samples_per_epoch["classifier"]
        avg_loss["classifier"] = running_loss["classifier"]/samples_per_epoch["classifier"]
        avg_loss["reconstruction"] = running_loss["reconstruction"]/(samples_per_epoch["classifier"])
        # print("Just finished step 2")
        conf_matrix["classifier"] = confusion_matrix(torch.cat(target_list["classifier"], dim=0).cpu().numpy(), torch.cat(prediction_list["classifier"], dim = 0).cpu().numpy())
        # print(conf_matrix["discriminator"])
        return accuracy, avg_loss, conf_matrix
    


    def validation_discriminator(self):

        self.encoder.eval()
        self.decoder.eval()
        self.classifier.eval()
        self.discriminator.eval()

        running_loss = {"discriminator": 0.0}

        samples_per_epoch = {"discriminator": 0}

        avg_loss = {"discriminator": 0.0}

        correct_predictions = {"discriminator":0}

        prediction_list = {"discriminator": []}

        target_list = {"discriminator": []}

        accuracy = {"discriminator": 0.0}

        conf_matrix = {"discriminator": None}

        softmax = nn.Softmax(dim = 1)
        sigmoid = nn.Sigmoid()

        with torch.no_grad():
            for discriminator_batch in self.val_dataloader_discriminator:
                inputs,targets = discriminator_batch
                inputs[0] = inputs[0].to(self.device).float()
                inputs[1] = inputs[1].to(self.device).float()

                discriminator_labels = targets.to(self.device)

                outputs_encoder_0, out_0= self.encoder(inputs[0])
                outputs_encoder_1, out_1 = self.encoder(inputs[1])


                inputs_discriminator = torch.cat([outputs_encoder_0, outputs_encoder_1], dim = 1).to(self.device)
                # inputs_discriminator = outputs_encoder_0 * outputs_encoder_1
                # inputs_discriminator = cosine_similarity(outputs_encoder_0, outputs_encoder_1)
                outputs_discriminator = self.discriminator(torch.squeeze(inputs_discriminator).detach())
                loss_discriminator = self.criterion_discriminator(outputs_discriminator, discriminator_labels)

                running_loss["discriminator"] += loss_discriminator.item()


                predictions_discriminator = torch.argmax(softmax(outputs_discriminator), dim = 1)

                # print(predictions_classifier_1)
                # print(activity_labels_1)
                

                correct_predictions["discriminator"] += (torch.squeeze(predictions_discriminator) == (discriminator_labels)).sum()

                prediction_list["discriminator"].append(torch.squeeze(predictions_discriminator))
                target_list["discriminator"].append(discriminator_labels)

                samples_per_epoch["discriminator"] += discriminator_labels.size(0)
        accuracy["discriminator"] = correct_predictions["discriminator"].item()/samples_per_epoch["discriminator"]
        avg_loss["discriminator"] = running_loss["discriminator"]/samples_per_epoch["discriminator"]
        # print("Just finished step 2")
        conf_matrix["discriminator"] = confusion_matrix(torch.cat(target_list["discriminator"], dim=0).cpu().numpy(), torch.cat(prediction_list["discriminator"], dim = 0).cpu().numpy())
        # print(conf_matrix["discriminator"])
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
                inputs_classifier, targets_classifier, participant = classifier_batch[0].to(self.device).float(), classifier_batch[1].to(self.device), classifier_batch[2]
                outputs_encoder_classifier,_ = self.encoder(inputs_classifier)
                validation_latent_space = validation_latent_space + transform_into_list_classifier(outputs_encoder_classifier, targets_classifier, participant)

        return validation_latent_space

        


    def train(self, epochs):
        early_stopping = EarlyStopping(patience=25, verbose=True, path='model_checkpoint.pt')
        for a in range(epochs):
            accuracy_train, avg_loss_train, confusion_matrix_train = self.train_one_epoch()
            print(f"||Training|| Acc: {accuracy_train} || Loss: {avg_loss_train} ||")
            accuracy_validation, avg_loss_validation, confusion_matrix_validation = self.validation_classifier()
            print(f"||Validation|| Acc: {accuracy_validation} || Loss: {avg_loss_validation} ||")
            early_stopping(accuracy_validation['classifier'], self.classifier)
            if (early_stopping.early_stop):
                print("Early stopping")
                break
        #     if a%5 == 0:
        #         train_latent, validation_latent =self.get_latent_discriminator()
        #         save_latent_space_discriminator(dataset = self.dataset, 
        #                           experiment = self.experiment,
        #                         seed = self.seed, data = train_latent, epoch = a, step = 2, model = "encoder",name = "train", self.id)
        #         save_latent_space_discriminator(dataset = self.dataset, 
        #                           experiment = self.experiment,
        #                         seed = self.seed, data = validation_latent, epoch = a, step = 2, model = "encoder",name = "validation", self.id)
        
        # val_classifier = self.get_latent_classifier()
        # save_latent_space_classifier(dataset = self.dataset, 
        #                           experiment = self.experiment,
        #                         seed = self.seed, data = val_classifier, epoch = a, step = 2, model = "classification",name = "validation", self.id)

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
        # print("Keys: %s" % list(file.keys()))

        for group in file.keys():
            # print(f"Reading {group}")

            grp = file[group]

            data_test = grp['data'][:]
            activity_label_test = grp['activity_label'][()]
            person_label_test = grp['person_label'][()]
            dataset_test.append((data_test, activity_label_test, person_label_test))
    return dataset_train, dataset_validation, dataset_test


def get_discriminator_dataset(path, trainsize):
    dataset_train, dataset_validation = [],[]
    with h5py.File(path+f"train.h5",'r') as file:
        # print("Keys: %s" % list(file.keys()))

        for group in file.keys():
            # print(f"Reading {group}")

            grp = file[group]

            # print(grp['data'])

            data_0_train = grp['data_0'][()]
            data_1_train = grp['data_1'][()]
            disc_label_train = grp['disc_label'][()]
            dataset_train.append(((data_0_train, data_1_train), disc_label_train))
    with h5py.File(path + f"validation.h5",'r') as file:
        # print("Keys: %s" % list(file.keys()))

        for group in file.keys():
            # print(f"Reading {group}")S
            grp = file[group]

            data_0_validation = grp['data_0'][()]
            data_1_validation = grp['data_1'][()]
            disc_label_validation = grp['disc_label'][()]
            dataset_validation.append(((data_0_validation, data_1_validation), disc_label_validation))
    return dataset_train, dataset_validation


def get_dataset_2(dataset, experiment, bs,PATH, trainsize):
    path_classification = PATH + f'datasets/{dataset}/prepared/{experiment}/classification/'
    path_discrimination = PATH + f'datasets/{dataset}/prepared/{experiment}/discrimination/'

    train_classifiaction, validation_classification, test_classification = get_classification_dataset(path_classification)
    train_discrimination, validation_discrimination = get_discriminator_dataset(path_discrimination, trainsize)

    final_train_discrimination, final_validation_discrimination = [],[]
    label_aux = []
    for pairs, label  in train_discrimination:
        final_train_discrimination.append(((train_classifiaction[pairs[0]][0],train_classifiaction[pairs[1]][0]), label))
        label_aux.append(label)
    for pairs, label  in validation_discrimination:
        final_validation_discrimination.append(((validation_classification[pairs[0]][0],validation_classification[pairs[1]][0]), label))

    print(np.unique(label_aux))

    train_dataloader_classification = DataLoader(train_classifiaction, 
                                  bs,shuffle=True, drop_last=False)
    validation_dataloader_classification = DataLoader(validation_classification,
                                       64, drop_last=True)
    test_dataloader_classification = DataLoader(test_classification,
                                       64, drop_last=True)
    
    number_batches_classification = len(train_dataloader_classification)
    bs_discriminator = len(train_discrimination) // number_batches_classification

    train_dataloader_discriminator =  DataLoader(final_train_discrimination, 
                                  bs_discriminator, shuffle=True, drop_last=True)
    validation_dataloader_discriminator = DataLoader(final_validation_discrimination, 
                                  128, drop_last=True)
    print(f"The number of samples for train classification: {len(train_dataloader_classification.dataset)}")
    print(f"The number of samples for validation classification {len(validation_dataloader_classification.dataset)}")
    print(f"The number of samples for test classification {len(test_dataloader_classification.dataset)}")
    print(f"The number of samples for train discriminator {len(train_dataloader_discriminator.dataset)}")
    print(f"The number of samples for validation discriminator {len(validation_dataloader_discriminator.dataset)}")

    # print(next(iter(train_dataloader_discriminator)))
    

    
    return train_dataloader_classification, validation_dataloader_classification, test_dataloader_classification, train_dataloader_discriminator, validation_dataloader_discriminator


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

    

    path_encoder = f"/models/Percom2024/{input_parameters['dataset']}/{input_parameters['experiment']}/{input_parameters['seed']}/step_1/encoder.pt"
    path_decoder = f"/models/Percom2024/{input_parameters['dataset']}/{input_parameters['experiment']}/{input_parameters['seed']}/step_1/decoder.pt"


    ##Load models    
    encoder = load_model(path_encoder)
    decoder = load_model(path_decoder)
    classifier = Classifier(sensors_channels[input_parameters['dataset']], input_parameters['seed'])
    discriminator = Discriminator_1()

    ##We get the dataloaders

    classification_train_dataloader, classification_validation_dataloader, classification_test_dataloader, discriminator_train_dataloader, discriminator_validation_dataloader = get_dataset(input_parameters['dataset'],
                                                                                                                                                                                             input_parameters['experiment'],
                                                                                                                                                                                             input_parameters['classifier_bs'])



    ## We define the loss functions

    criterion = {"reconstruction": nn.MSELoss(),
    "discriminator": nn.CrossEntropyLoss(),
    "classifier": nn.CrossEntropyLoss()}

    ## We define the optimizers
    
    optimizer = {"encoder": optim.Adam(encoder.parameters() , lr= input_parameters['decoder_lr'], weight_decay= 10e-5),
    "decoder": optim.Adam(decoder.parameters(), lr= input_parameters['decoder_lr'], weight_decay= 10e-5),
    "discriminator": optim.Adam(discriminator.parameters(), lr = input_parameters['discriminator_lr']),
    "classifier": optim.Adam(classifier.parameters(), lr= input_parameters['classifier_lr'], weight_decay= 10e-5)}

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
                              input_parameters['dataset'] )
    
    encoder, decoder, discriminator, classifier = trainerB.train(30)

    path = f'/models/Percom2024/{input_parameters["dataset"]}/{input_parameters["experiment"]}/{input_parameters["seed"]}/step_2/'

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