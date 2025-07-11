import torch
import torch.nn
import os
import sys
sys.path.append(os.getcwd())
from utils.model.feature_extractor import Feature_Extractor
import h5py

def transform_into_list_classifier(outputs, labels, participant):
    tensor = outputs.detach()
    data = [(a,b,c) for a,b,c in zip(tensor,labels,participant)]
    return data

def save_latent_space_classifier(dataset, experiment, seed, data, epoch, step, model,name,id):
    path = f"/checkpoints/Percom2024/{dataset}/{experiment}/{seed}/{id}/step_{step}/{model}/{epoch}/"

    if not os.path.exists(path):
    # Create the directory
        os.makedirs(path)
        print(f'Directory {path} created')
    else:
        print(f'Directory {path} already exists')

    with h5py.File(path+f'{name}_latent_space.h5', 'w') as hf:
                for i, (data, disc_label, participant) in enumerate(data):
                    grp = hf.create_group(f'item_{i}')
                    grp.create_dataset('data', data=data.cpu().numpy())
                    grp.create_dataset('activity', data=disc_label.cpu().numpy())
                    grp.create_dataset('participant', data=participant.cpu().numpy())
    

 

if __name__ == "__main__":
    dummy_model = Feature_Extractor(input_shape = (64,1,128) , num_blocks = [2,2,0,0], in_channel = 3, seed = 0)
    dummy_input = torch.rand((32,3,512))
    dummy_labels = torch.randint(0,4,(32,))
    dummy_output = dummy_model(dummy_input)

    data = transform_into_list(dummy_output, dummy_labels)
    data_1 = transform_into_list(dummy_output, dummy_labels)

    data_final = []

    data_final = data_final + data
    data_final = data_final + data_1

    # print(len(data_final))



