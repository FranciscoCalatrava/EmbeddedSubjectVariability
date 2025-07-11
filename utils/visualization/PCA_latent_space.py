import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import h5py
import plotly.express as px
from torch.utils.tensorboard import SummaryWriter



def get_dataset(dataset, experiment, seed, step, model, epoch):
    path_train = f"./checkpoints/{dataset}/{experiment}/{seed}/step_{step}/{model}/{epoch}/train_latent_space.h5"
    path_validation = f"./checkpoints/{dataset}/{experiment}/{seed}/step_{step}/{model}/{epoch}/validation_latent_space.h5"

    dataset_train, dataset_validation, dataset_images = [],[],[]
    # with h5py.File(path_train,'r') as file:
    #     # print("Keys: %s" % list(file.keys()))

    #     for group in file.keys():
    #         # print(f"Reading {group}")

    #         grp = file[group]

    #         # print(grp['data'])

    #         data_train = grp['data'][:]
    #         disc_label = grp['disc_label'][()]
    #         dataset_train.append((data_train, disc_label))
    
    with h5py.File(path_validation,'r') as file:
        print("Keys: %s" % list(file.keys()))

        for group in file.keys():
            # print(f"Reading {group}")

            grp = file[group]

            data_validation = grp['data'][:]
            activity = grp['activity'][()]
            participant = grp['participant'][()]
            dataset_validation.append((data_validation, activity))

    return dataset_train, dataset_validation


if __name__ == "__main__":

    input_parameters = {"dataset":sys.argv[1], "experiment": sys.argv[2], "seed": sys.argv[3]}
    train, validation = get_dataset(input_parameters['dataset'],
                                    input_parameters['experiment'],
                                    input_parameters['seed'],
                                    2,
                                    "classification",
                                    40)
    
    writer = SummaryWriter('runs/visualization_1')

    
    

    # features,labels = [],[]
    
    # for a,b in train:
    #     features.append(a)
    #     labels.append(b)

    # # Standardize the latent vectors
    # scaler = StandardScaler()
    # latent_vectors_scaled = scaler.fit_transform(np.vstack(features))

    # # Perform PCA
    # pca = PCA(n_components=3)
    # principal_components = pca.fit_transform(latent_vectors_scaled)

    # # Create a DataFrame for visualization
    # df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    # df['Label'] = np.array(labels)
    # fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Label',
    #                     title='3D PCA of Latent Space',
    #                     labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'},
    #                     color_continuous_scale='viridis')

    # # Show the plot in an interactive window
    # fig.show()

    features,labels = [],[]

    for a,b in validation:
        features.append(a)
        labels.append(b)

    # # Standardize the latent vectors
    # scaler = StandardScaler()
    # latent_vectors_scaled = scaler.fit_transform(np.vstack(features))

    # # Perform PCA
    # pca = PCA(n_components=3)
    # principal_components = pca.fit_transform(latent_vectors_scaled)



    writer.add_embedding(np.vstack(features),
                    metadata=np.vstack(labels))
    writer.close()

    # tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=3000)
    # tsne_results = tsne.fit_transform(np.vstack(features))

    # # Create a DataFrame for visualization
    # df = pd.DataFrame(data=tsne_results, columns=['PC1', 'PC2'])
    # df['Label'] = np.array(labels)

    # # Create an interactive scatter plot with Plotly
    # fig = px.scatter(df, x='PC1', y='PC2', color='Label',
    #                 title='3D PCA of Latent Space',
    #                 labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
    #                 color_continuous_scale='viridis')

    # # Show the plot in an interactive window
    # fig.show()

