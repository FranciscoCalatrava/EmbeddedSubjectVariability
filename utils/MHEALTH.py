import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter

class MHEALTH():
    def __init__(self, train, validation, test, PATH):
        self.train_participant = train
        self.validation_participant = validation
        self.test_participant = test

        self.training = None
        self.test = None
        self.validation = None

        self.training_cleaned = None
        self.test_cleaned = None
        self.validation_cleaned = None

        self.training_normalized = None
        self.test_normalized = None
        self.validation_normalized = None

        self.training_normalized_segmented = None
        self.test_normalized_segmented = None
        self.validation_normalized_segmented = None

        self.training_final = None
        self.validation_final = None
        self.test_final = None


        self.headers = [
    "chest_sensor_acceleration_x",
    "chest_sensor_acceleration_y",
    "chest_sensor_acceleration_z",
    "ecg_lead_1",
    "ecg_lead_2",
    "left_ankle_sensor_acceleration_x",
    "left_ankle_sensor_acceleration_y",
    "left_ankle_sensor_acceleration_z",
    "left_ankle_sensor_gyro_x",
    "left_ankle_sensor_gyro_y",
    "left_ankle_sensor_gyro_z",
    "left_ankle_sensor_magnetometer_x",
    "left_ankle_sensor_magnetometer_y",
    "left_ankle_sensor_magnetometer_z",
    "right_lower_arm_sensor_acceleration_x",
    "right_lower_arm_sensor_acceleration_y",
    "right_lower_arm_sensor_acceleration_z",
    "right_lower_arm_sensor_gyro_x",
    "right_lower_arm_sensor_gyro_y",
    "right_lower_arm_sensor_gyro_z",
    "right_lower_arm_sensor_magnetometer_x",
    "right_lower_arm_sensor_magnetometer_y",
    "right_lower_arm_sensor_magnetometer_z",
    "activityID"
]
        self.PATH = PATH
    def get_datasets(self):
        training = {a:0 for a in self.train_participant}
        test = {a:0 for a in self.test_participant}
        validation ={a:0 for a in self.validation_participant}

        print(training)
        
        for b in training.keys():
            data = pd.read_csv(self.PATH + f"datasets/MHEALTH/normal/mHealth_subject{b}.log", sep= '\t')
            data.columns = self.headers
            training[b] = data
        for b in validation.keys():
            data = pd.read_csv(self.PATH + f"datasets/MHEALTH/normal/mHealth_subject{b}.log", sep= '\t')
            data.columns = self.headers
            validation[b] = data
        for b in test.keys():
            data = pd.read_csv(self.PATH+ f"datasets/MHEALTH/normal/mHealth_subject{b}.log", sep= '\t')
            data.columns = self.headers
            test[b] = data
        
        self.training = training
        self.test = test
        self.validation = validation

    def normalize(self):
        training_normalized = {a:0 for a in self.training_cleaned.keys()}
        test_normalized = {a:0 for a in self.test_cleaned.keys()}
        validation_normalized ={a:0 for a in self.validation_cleaned.keys()}

        max = pd.DataFrame(np.zeros((1,len(self.headers))), columns= self.headers)
        min = pd.DataFrame(np.zeros((1,len(self.headers))), columns= self.headers)

        min_aux, max_aux = None, None


        for a in training_normalized.keys():
            max_aux = self.training_cleaned[a].max(axis = 'rows')
            min_aux = self.training_cleaned[a].min(axis = 'rows')

            for indx, a in enumerate(max):
                if max.iloc[0,indx] < max_aux.iloc[indx]:
                     max.iloc[0,indx] = max_aux.iloc[indx]
                if min.iloc[0,indx] > min_aux.iloc[indx]:
                     min.iloc[0,indx] = min_aux.iloc[indx]        
        for a in training_normalized.keys():
            training_normalized[a] = pd.DataFrame(((self.training_cleaned[a].values - min.values)/(max.values- min.values)), columns= self.headers)
            training_normalized[a]["activityID"] = self.training_cleaned[a]["activityID"]        
        for a in test_normalized.keys():
            test_normalized[a] = pd.DataFrame((self.test_cleaned[a].values - min.values)/(max.values- min.values), columns= self.headers)
            test_normalized[a]["activityID"] = self.test_cleaned[a]["activityID"]
        for a in validation_normalized.keys():
            validation_normalized[a] = pd.DataFrame(((self.validation_cleaned[a].values - min.values)/(max.values- min.values)), columns= self.headers)
            validation_normalized[a]["activityID"] = self.validation_cleaned[a]["activityID"]

        self.training_normalized = training_normalized
        self.test_normalized = test_normalized
        self.validation_normalized = validation_normalized

        print(validation_normalized)

    def segment_data(self, data_dict, window_size, overlap):
        """
        Segments the data into fixed-width windows with overlapping.

        :param data_dict: Dictionary with participant ID as keys and DataFrames as values.
        :param window_size: The size of each window (number of rows).
        :param overlap: The overlap between consecutive windows (number of rows).
        :return: A dictionary with the same keys as data_dict and values as lists of segmented DataFrames.
        """
        segmented_data = {}

        for participant_id, df in data_dict.items():
            num_rows = len(df)
            segments = []
            start = 0
            while start < num_rows:
                end = start + window_size
                if end > num_rows:
                    break
                segment = df.iloc[start:end,:]
                # Check if the segment contains more than one unique label, if so, skip this segment
                if len(segment.iloc[:, -1].unique()) > 1:
                    start += 1
                    continue
                segments.append(segment)
                start += overlap
            segmented_data[participant_id] = segments
        return segmented_data
    
    def clean_nan(self, data):
        data_clean = {a:0 for a in data.keys()}
        for a in data.keys():
            data_aux =  data[a].ffill(axis = 0).bfill(axis = 0)
            data_clean[a] = data_aux
        return data_clean

    def butter_lowpass(self,cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    def preprocessing(self):
        training_cleaned_aux = self.clean_nan(self.training)
        test_cleaned_aux = self.clean_nan(self.test)
        validation_cleaned_aux = self.clean_nan(self.validation)


        for a in training_cleaned_aux.keys():
            training_cleaned_aux[a] = training_cleaned_aux[a][training_cleaned_aux[a]["activityID"] !=0 ]
            training_cleaned_aux[a].reset_index(drop=True, inplace=True)
        for a in validation_cleaned_aux.keys():
            validation_cleaned_aux[a] = validation_cleaned_aux[a][validation_cleaned_aux[a]["activityID"] !=0 ]
            validation_cleaned_aux[a].reset_index(drop=True, inplace=True)
        for a in test_cleaned_aux.keys():
            test_cleaned_aux[a] = test_cleaned_aux[a][test_cleaned_aux[a]["activityID"] !=0 ]
            test_cleaned_aux[a].reset_index(drop=True, inplace=True)

        # exclude_columns = ['activityID']

        # for a in training_cleaned_aux.keys():
        #     for col in training_cleaned_aux[a].columns:
        #         if col not in exclude_columns:
        #             training_cleaned_aux[a][col] = self.butter_lowpass_filter(training_cleaned_aux[a][col], 15, 50, 4)
        
        # for a in test_cleaned_aux.keys():
        #     for col in test_cleaned_aux[a].columns:
        #         if col not in exclude_columns:
        #             test_cleaned_aux[a][col] = self.butter_lowpass_filter(test_cleaned_aux[a][col], 15, 50, 4)

        # for a in validation_cleaned_aux.keys():
        #     for col in validation_cleaned_aux[a].columns:
        #         if col not in exclude_columns:
        #             validation_cleaned_aux[a][col] = self.butter_lowpass_filter(validation_cleaned_aux[a][col], 15, 50, 4)
        self.training_cleaned = training_cleaned_aux
        self.test_cleaned = test_cleaned_aux
        self.validation_cleaned = validation_cleaned_aux


    def data_segmentation(self):
        train_data_segmented = self.segment_data(self.training_cleaned, 512, 256)
        validation_data_segmented = self.segment_data(self.validation_cleaned, 512, 256)
        test_data_segmented = self.segment_data(self.test_cleaned, 512, 256)

        self.training_normalized_segmented = train_data_segmented
        self.test_normalized_segmented = test_data_segmented
        self.validation_normalized_segmented = validation_data_segmented

    def prepare_dataset(self):

        training, validation, testing = [], [], []
        signals = np.array([0,1,2,5,6,7,8,9,10,14,15,16,17,18,19])

        for a in self.training_normalized_segmented.keys():
            for b in self.training_normalized_segmented[a]:
                training.append((np.transpose(b.iloc[:,signals].to_numpy()), int(b.iloc[0,-1]-1), int(a)))
        for a in self.validation_normalized_segmented.keys():
            for b in self.validation_normalized_segmented[a]:
                validation.append((np.transpose(b.iloc[:,signals].to_numpy()), int(b.iloc[0,-1]-1), int(a)))
        for a in self.test_normalized_segmented.keys():
            for b in self.test_normalized_segmented[a]:
                testing.append((np.transpose(b.iloc[:,signals].to_numpy()), int(b.iloc[0,-1])-1, int(a)))
        
        self.training_final = training
        self.validation_final = validation
        self.testing_final = testing