import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import f1_score

class Tester:
    def __init__(self, encoder, classifier, device):
        self.encoder = encoder
        self.classifier = classifier
        self.device = device

    def test(self,dataloader):
        self.encoder.eval()  # set the model to evaluation mode
        self.classifier.eval()
        correct_predictions = 0
        total_samples = 0
        targets_list = []
        predictions_list = []

        with torch.no_grad():
            for data in dataloader:
                inputs, targets = data[0].to(self.device).float(), data[1].to(self.device)
                #print(inputs.shape)
                outputs_1 = self.classifier(self.encoder(inputs))
                predicted_classes = outputs_1.argmax(dim=1).squeeze()  # Find the class index with the maximum value in predicted
                correct_predictions += (predicted_classes == targets).sum().float()
                total_samples += targets.size(0)
                targets_list.append(targets)
                predictions_list.append(predicted_classes)
        cm = confusion_matrix(torch.cat(targets_list, dim=0).cpu().numpy(), torch.cat(predictions_list, dim = 0).cpu().numpy())
        # np.savetxt("./confusion/confusion_matrix.txt", cm, fmt="%d", newline='\n')
        #print("Total samples testing: ",total_samples)
        accuracy = correct_predictions / total_samples
        F1_macro = f1_score(torch.cat(targets_list, dim=0).cpu().numpy(), torch.cat(predictions_list, dim = 0).cpu().numpy(), average='macro')
        F1_w = f1_score(torch.cat(targets_list, dim=0).cpu().numpy(), torch.cat(predictions_list, dim = 0).cpu().numpy(), average='weighted')

        print(f"Test F1: {F1_macro:.4f}")
        return accuracy, F1_macro, F1_w