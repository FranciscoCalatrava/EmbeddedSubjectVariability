import numpy as np
import pandas as pd
import torch
import os
import sys
import itertools
from itertools import combinations, product
import random



def Same_Person_Same_Activity(dataset, id_participants, id_classes, dictionary_dataset):

    same_person_same_activity = {}
    counter = 0


    for a in id_classes:
        aux_dictionary = {}
        for b in id_participants:
            aux_dictionary[b] = list(itertools.combinations(dictionary_dataset[b][a],2))
        same_person_same_activity[a] = aux_dictionary

    counter = 0
    final_same_person_same_activity = {}
    for a in id_classes :
        aux= []
        for b in id_participants:
            for c in same_person_same_activity[a][b]:
                aux.append(c)
        final_same_person_same_activity[a] = aux

    tester = []

    for a in final_same_person_same_activity.keys():
        for b in final_same_person_same_activity[a]:
            tester.append((dataset[b[0]][1] == dataset[b[1]][1]) and (dataset[b[0]][2] == dataset[b[1]][2]))
    print("Same Person Same Activity",set(tester))

    return final_same_person_same_activity

def Different_Person_Same_Activity(dataset, id_participants, id_classes, dictionary_dataset):

    different_person_same_activity = {}
    vector_of_participants = id_participants
    pairs_of_participants = list(itertools.combinations(vector_of_participants,2))
    counter = 0

    # theoretical_numer_of_samples_different_person_same_activity = 0



    for a in id_classes:
        aux_dictionary = {}
        for b in pairs_of_participants:
            aux_dictionary[counter] = list(itertools.product(dictionary_dataset[b[0]][a],dictionary_dataset[b[1]][a]))
            # theoretical_numer_of_samples_different_person_same_activity += (count_samples[a[0]][b]*count_samples[a[1]][b])
            counter += 1
        different_person_same_activity[a] = aux_dictionary


    # practical_number_of_samples_different_person_same_activity = 0
    # for a in different_person_same_activity.keys():
    #     practical_number_of_samples_different_person_same_activity += len(different_person_same_activity[a])
    #     # print(len(different_person_same_activity[a]))

    # print(practical_number_of_samples_different_person_same_activity)


    final_different_person_same_activity = {}
    counter = 0
    for a in id_classes:
        aux = []
        for b in pairs_of_participants:
            for c in different_person_same_activity[a][counter]:
                aux.append(c)
            final_different_person_same_activity[a] =  aux
            counter += 1
    
    # print(final_different_person_same_activity)


    tester = []

    for a in final_different_person_same_activity.keys():
        for b in final_different_person_same_activity[a]:
            tester.append((dataset[b[0]][1] == dataset[b[1]][1]) and (dataset[b[0]][2] != dataset[b[1]][2]))
    print("Different Person Same Activity", set(tester))

    return final_different_person_same_activity

def Same_Person_Different_Activity(dataset, id_participants, id_classes, dictionary_dataset):

    same_person_different_activity = {}

    pairs_of_classes = list(combinations(id_classes,2))
    counter = 0
    # theoretical_number_of_samples_same_person_different_activity = 0

    for a in id_participants:
        for b in pairs_of_classes:
            same_person_different_activity[counter] = list(product(dictionary_dataset[a][b[0]], dictionary_dataset[a][b[1]]))
            # theoretical_number_of_samples_same_person_different_activity += (count_samples[a][b[0]]*count_samples[a][b[1]])
            counter += 1

    final_same_person_different_activity = {}
    counter = 0
    for a in id_participants:
        aux = []
        for b in pairs_of_classes:
            for c in same_person_different_activity[counter]:
                aux.append(c)
                final_same_person_different_activity[a] =  aux
            counter +=1

  

    
    practical_number_of_samples_same_person_different_activity = 0
    for a in same_person_different_activity.keys():
        # print(len(same_person_different_activity[a]))
        practical_number_of_samples_same_person_different_activity += len(same_person_different_activity[a])
    # print(count_samples)
    # print(theoretical_number_of_samples_same_person_different_activity)
    print(practical_number_of_samples_same_person_different_activity)


    tester = []

    for a in final_same_person_different_activity.keys():
        for b in final_same_person_different_activity[a]:
            tester.append((dataset[b[0]][1] != dataset[b[1]][1]) and (dataset[b[0]][2] == dataset[b[1]][2]))
    print("Same Person Different Activity", set(tester))


    return final_same_person_different_activity

def Different_Person_Different_Activity(dataset, id_participants, id_classes, dictionary_dataset):

    different_person_different_activity = {}
    counter = 0
    pairs_of_classes = list(combinations(id_classes,2))
    pairs_of_participants = list(combinations(id_participants,2))

    # theoretical_number_of_samples_different_person_different_activity = 0

    for a in pairs_of_participants:
        for b in pairs_of_classes:
            different_person_different_activity[counter] = list(product(dictionary_dataset[a[0]][b[0]], dictionary_dataset[a[1]][b[1]]))
            # theoretical_number_of_samples_different_person_different_activity += (count_samples[a[0]][b[0]]*count_samples[a[1]][b[1]])
            counter +=1

    final_different_person_different_activity = {}
    counter  =0

    for a in pairs_of_participants:
        aux = []
        for b in pairs_of_classes:
            for c in different_person_different_activity[counter]:
                aux.append(c)
                final_different_person_different_activity[a] =  aux
            counter +=1

    # counter = 0
    practical_number_of_samples_different_person_different_activity = 0
    for a in different_person_different_activity.keys():
        # print(len(different_person_different_activity[a]))
        practical_number_of_samples_different_person_different_activity += len(different_person_different_activity[a])
        # counter +=1

    # print(counter)


    # print(different_person_same_activity)
    print(practical_number_of_samples_different_person_different_activity)
    # print(theoretical_number_of_samples_different_person_different_activity)

    tester = []
    for a in final_different_person_different_activity.keys():
        for b in final_different_person_different_activity[a]:
            tester.append((dataset[b[0]][1] != dataset[b[1]][1]) and (dataset[b[0]][2] != dataset[b[1]][2]))
    print("Different Person Different Activity", set(tester))

    return final_different_person_different_activity



def get_minimum_number_of_samples(datasets):
    samples_count = []
    for dataset in datasets:
        for key in dataset.keys():
            samples_count.append(len(dataset[key]))
    return min(samples_count)




def sample_dataset(data, percentage, minimum):
    number_of_samples_per_pair = int(percentage*minimum)

    number_keys = len(data.keys())
    num_elements = number_of_samples_per_pair

    selected = {}
    for key, value in data.items():
        if len(value) >= num_elements:
            selected[key] = random.sample(value, num_elements)
        else:
            selected[key] = value
    return selected

def construct_dataset(dataset, num_samples):
    ## -------------------Inputs-----------------#
    ## dataset: List with raw signals, activity label, and participant label [(signals, A_label, P_label)_0,.....(signals, A_label, P_label)_N]
    ## -------------------Outputs----------------#
    ## dataset_discriminator: List with following structur ((pair_0, pair_1),(D_label, A_label_0, A_label_1))
    
    classes_index, participants_index = [],[]

    for a in dataset:
        classes_index.append(a[1])
        participants_index.append(a[2])
    
    id_classes = np.unique(classes_index)
    id_participants = np.unique(participants_index)


    index_dataset = []

    ##We create a new dataset replacing the data by the index of this data structure

    for indx, data in enumerate(dataset):
        index_dataset.append((indx, data[1], data[2]))

    count_samples = {a:{b:0 for b in id_classes} for a in id_participants}
    dictionary_dataset = {a:{b:[] for b in id_classes} for a in  id_participants}

    for a in index_dataset:
        dictionary_dataset[a[2]][a[1]].append(a[0])
    
    for a in dataset:
        count_samples[a[2]][a[1]] = count_samples[a[2]][a[1]] +1

    same_person_same_activity           = Same_Person_Same_Activity(dataset, id_participants, id_classes, dictionary_dataset)
    different_person_same_activity      = Different_Person_Same_Activity(dataset, id_participants, id_classes, dictionary_dataset)
    same_person_different_activity      = Same_Person_Different_Activity(dataset, id_participants, id_classes, dictionary_dataset)
    different_person_different_activity = Different_Person_Different_Activity(dataset, id_participants, id_classes, dictionary_dataset)


    sampled_same_person_same_activity            = sample_dataset(same_person_same_activity, num_samples)
    sampled_different_person_same_activity       = sample_dataset(different_person_same_activity, num_samples)
    sampled_same_person_different_activity       = sample_dataset(same_person_different_activity,num_samples)
    sampled_different_person_different_activity  = sample_dataset(different_person_different_activity,num_samples)

    dataset_final = []

    for a in sampled_same_person_same_activity.keys():
        for b in sampled_same_person_same_activity[a]:
            dataset_final.append(((b[0], b[1]),1))
    for a in sampled_different_person_same_activity.keys():
        for b in sampled_different_person_same_activity[a]:
            dataset_final.append(((b[0], b[1]),0))
    # for a in sampled_same_person_different_activity.keys():
    #     for b in sampled_same_person_different_activity[a]:
    #         dataset_final.append(((b[0], b[1]),3))
    # for a in sampled_different_person_different_activity.keys():
    #     for b in sampled_different_person_different_activity[a]:
    #         dataset_final.append(((b[0], b[1]),2))
    return dataset_final


def construct_dataset_DCGAN(dataset, num_samples):
    ## -------------------Inputs-----------------#
    ## dataset: List with raw signals, activity label, and participant label [(signals, A_label, P_label)_0,.....(signals, A_label, P_label)_N]
    ## -------------------Outputs----------------#
    ## dataset_discriminator: List with following structur ((pair_0, pair_1),(D_label, A_label_0, A_label_1))
    
    classes_index, participants_index = [],[]

    for a in dataset:
        classes_index.append(a[1])
        participants_index.append(a[2])
    
    id_classes = np.unique(classes_index)
    id_participants = np.unique(participants_index)


    index_dataset = []

    ##We create a new dataset replacing the data by the index of this data structure

    for indx, data in enumerate(dataset):
        index_dataset.append((indx, data[1], data[2]))

    count_samples = {a:{b:0 for b in id_classes} for a in id_participants}
    dictionary_dataset = {a:{b:[] for b in id_classes} for a in  id_participants}

    for a in index_dataset:
        dictionary_dataset[a[2]][a[1]].append(a[0])
    
    for a in dataset:
        count_samples[a[2]][a[1]] = count_samples[a[2]][a[1]] +1

    same_person_same_activity           = Same_Person_Same_Activity(dataset, id_participants, id_classes, dictionary_dataset)
    different_person_same_activity      = Different_Person_Same_Activity(dataset, id_participants, id_classes, dictionary_dataset)
    same_person_different_activity      = Same_Person_Different_Activity(dataset, id_participants, id_classes, dictionary_dataset)
    different_person_different_activity = Different_Person_Different_Activity(dataset, id_participants, id_classes, dictionary_dataset)

    minimum = get_minimum_number_of_samples([same_person_same_activity, different_person_same_activity])
    print(f"THe minimum number of samples is: {minimum}")
    sampled_same_person_same_activity            = sample_dataset(same_person_same_activity, num_samples)
    sampled_different_person_same_activity       = sample_dataset(different_person_same_activity, num_samples)
    sampled_same_person_different_activity       = sample_dataset(same_person_different_activity,num_samples)
    sampled_different_person_different_activity  = sample_dataset(different_person_different_activity,num_samples)

    dataset_final_positive = []
    dataset_final_negative = []

    for a in sampled_same_person_same_activity.keys():
        for b in sampled_same_person_same_activity[a]:
            dataset_final_positive.append(((b[0], b[1]),1))
    for a in sampled_different_person_same_activity.keys():
        for b in sampled_different_person_same_activity[a]:
            dataset_final_negative.append(((b[0], b[1]),0))
    # for a in sampled_same_person_different_activity.keys():
    #     for b in sampled_same_person_different_activity[a]:
    #         dataset_final.append(((b[0], b[1]),3))
    # for a in sampled_different_person_different_activity.keys():
    #     for b in sampled_different_person_different_activity[a]:
    #         dataset_final.append(((b[0], b[1]),2))

    print(f"The number of samples in positive dataset is: {len(dataset_final_positive)}")
    print(f"The number of samples in negative dataset is: {len(dataset_final_negative)}")
    return dataset_final_positive, dataset_final_negative