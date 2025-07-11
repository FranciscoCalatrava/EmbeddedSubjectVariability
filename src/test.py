import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import f1_score
import os

class Tester:
    def __init__(self, encoder, classifier, device):
        self.encoder = encoder
        self.classifier = classifier
        self.device = device

    def test(self,dataloader, dataset, participant):
        self.encoder.eval()  # set the model to evaluation mode
        self.classifier.eval()
        correct_predictions = 0
        total_samples = 0
        targets_list = []
        predictions_list = []
        pid = "not_pid"

        with torch.no_grad():
            for data in dataloader:
                inputs, targets = data[0].to(self.device).float(), data[1].to(self.device)
                #print(inputs.shape)
                encoder,_ = self.encoder(inputs)
                outputs_1 = self.classifier(encoder)
                predicted_classes = outputs_1.argmax(dim=1).squeeze()  # Find the class index with the maximum value in predicted
                correct_predictions += (predicted_classes == targets).sum().float()
                total_samples += targets.size(0)
                targets_list.append(targets)
                predictions_list.append(predicted_classes)
        pid = str(os.getpid())            
        cm = confusion_matrix(torch.cat(targets_list, dim=0).cpu().numpy(), torch.cat(predictions_list, dim = 0).cpu().numpy())
        data_to_save = np.concatenate((torch.cat(targets_list, dim=0).cpu().numpy()[:,np.newaxis],torch.cat(predictions_list, dim = 0).cpu().numpy()[:,np.newaxis]), axis=1)
        # np.savetxt(f'./confusion/confusion_matrix_{dataset}_{participant}_' + pid +".txt", cm, fmt="%d", newline='\n')
        # np.savetxt(f'./confusion/prediction_data_{dataset}_{participant}_' + pid +".txt", data_to_save, fmt="%d", newline='\n')
        print("Total samples testing: ",total_samples)
        accuracy = correct_predictions / total_samples
        F1_macro = f1_score(torch.cat(targets_list, dim=0).cpu().numpy(), torch.cat(predictions_list, dim = 0).cpu().numpy(), average='macro')
        F1_w = f1_score(torch.cat(targets_list, dim=0).cpu().numpy(), torch.cat(predictions_list, dim = 0).cpu().numpy(), average='weighted')

        print(f"Test F1: {F1_macro:.4f}")
        return accuracy, F1_macro, F1_w
