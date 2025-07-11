# ╔═════════════════════════════════╗
# ║ Environment Variable Integration ║
# ╚═════════════════════════════════╝
import os
import sys

# Replace 'ENV_VAR_NAME' with the actual name of your environment variable
DATA_PATH = os.getenv('CONTRASTIVE_DATA_ROOT')
ROOT_PATH = os.getenv('CONTRASTIVE_ROOT')

if ROOT_PATH is not None:
    sys.path.append(ROOT_PATH)
else:
    print("Environment variable not found.")

# ╔═════════════════════════════════╗
# ║ Other Packages initialization   ║
# ╚═════════════════════════════════╝

import torch
import numpy as np
import datetime
import h5py
from torch.utils.data import DataLoader
import random
import yaml
import importlib
import pkgutil
import torch
import copy

# ╔═════════════════════════════════╗
# ║ Functions                       ║
# ╚═════════════════════════════════╝

import torch

def quaternion_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Perform quaternion multiplication q1 * q2.
    Both q1 and q2 are of shape (..., 4), representing (w, x, y, z).
    Returns the product of shape (..., 4).
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    # Quaternion product
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack([w, x, y, z], dim=-1)


def cycle_consistency_loss(q_pred: torch.Tensor,
                           q_pred_rot: torch.Tensor,
                           q_delta: torch.Tensor) -> torch.Tensor:
    """
    Enforces q_pred_rot ≈ q_delta * q_pred in quaternion space.
    
    Args:
      q_pred:       (N, 4) unit quaternions predicted for original sample x
      q_pred_rot:   (N, 4) unit quaternions predicted for rotated sample x'
      q_delta:      (N, 4) unit quaternions of the known rotation from x to x'
    
    Returns:
      Scalar loss. Minimizing it drives q_pred_rot close to q_delta*q_pred
      up to a sign ambiguity (±q). We use absolute dot product to handle that.
    """
    # Compose the predicted orientation for x with the known rotation Δ
    # => we get the expected orientation for x'.
    q_compose = quaternion_mul(q_delta, q_pred)  # shape (N, 4)

    # Dot product between q_pred_rot and (q_delta * q_pred)
    # sign invariance => we take the absolute value.
    dotval = torch.sum(q_pred_rot * q_compose, dim=1).abs()
    
    # A typical form: loss = 1 - mean(dot).
    # If dot=1 => perfect alignment => loss=0.
    # If dot=0 => 90° off => loss=1. 
    loss = 1.0 - dotval.mean()
    return loss




def import_dataset_classes(package_name):
    """
    Dynamically import all modules from the specified package and return a dictionary mapping
    the module name (or a derived key) to the dataset class found in that module.
    """
    dataset_classes = {}
    # Import the package itself
    package = importlib.import_module(package_name)

    # Iterate over all modules in the package directory
    for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
        if is_pkg:
            continue  # Skip subpackages

        full_module_name = f"{package_name}.{module_name}"
        module = importlib.import_module(full_module_name)

        # Assume the dataset class name is the same as the module name.
        dataset_class = getattr(module, module_name, None)

        # If not found, try capitalizing the module name
        if dataset_class is None:
            dataset_class = getattr(module, module_name.capitalize(), None)

        if dataset_class is not None:
            dataset_classes[module_name] = dataset_class
            print(f"Imported dataset class '{dataset_class.__name__}' from module '{full_module_name}'")
        else:
            print(f"[Warning] Could not find a dataset class in module '{full_module_name}'")

    return dataset_classes


def set_random_seed(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Function to print messages with a decorative header, footer, and timestamp
def fancy_print(message):
    current_time = datetime.datetime.now()  # Get current time
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")  # Format current time as string
    print("\n" + "=" * 40)
    print(f"{timestamp} - {message}")
    print("=" * 40 + "\n")


def get_data(dataset_instance):
    """
    Calls the necessary methods on the dataset instance to generate training, validation, and test splits.
    """
    if dataset_instance is None:
        raise ValueError("Empty dataset instance")
    else:
        dataset_instance.get_datasets()
        dataset_instance.preprocessing()
        dataset_instance.normalize()
        dataset_instance.data_segmentation()
        dataset_instance.prepare_dataset()

        # Expecting that these attributes are set by prepare_dataset()
        train = [(a[0], a[1], a[2]) for a in dataset_instance.training_final]
        validation = [(a[0], a[1], a[2]) for a in dataset_instance.validation_final]
        test = [(a[0], a[1], a[2]) for a in dataset_instance.testing_final]


        fancy_print(f"The length of the training data is {len(train)}")
        fancy_print(f"The length of the validation data is {len(validation)}")
        fancy_print(f"The length of the testing data is {len(test)}")

    return {"Train": train, "Validation": validation, "Test": test}


def get_dataloader(args):
    LOSO_DISTRIBUTION = None
    set_random_seed(args.seed)

    try:
        # Load LOSO distributions (if needed for future use)
        with open("./LOSO_DISTRIBUTIONS.yaml", "r") as f:
            LOSO_DISTRIBUTION = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] We failed loading the distribution file. Check if it exists")

    LOSO_DISTRIBUTION = LOSO_DISTRIBUTION.get(args.dataset)[f'{args.experiment}']
    

    dataset_classes = import_dataset_classes("dataset")
    dataset_instance = dataset_classes[args.dataset](
        train=LOSO_DISTRIBUTION.get('train'),
        validation=LOSO_DISTRIBUTION.get('validation'),
        test=LOSO_DISTRIBUTION.get('test'),
        current_directory=DATA_PATH
    )
    print("\nDataset instances created:")
    print(dataset_instance)
    data_distributions = get_data(dataset_instance)
    DATALOADERS = {
        'Train': DataLoader(data_distributions['Train'], batch_size=args.bs, shuffle=True, drop_last=True),
        'Validation': DataLoader(data_distributions['Validation'], shuffle = True, batch_size=args.bs, drop_last=True),
        'Test': DataLoader(data_distributions['Test'], batch_size=args.bs)
    }

    return DATALOADERS

def _get_sensors_indices(dataset):
    datasets = {
        "DSADS": {
            # DSADS has five IMU locations. Each one has 3 sensor types (acc, gyro, mag)
            # with 3 axes each → 9 columns per IMU.
            "T": list(range(0, 6)),  # Indices 0-8 for T (trunk)
            "RA": list(range(6, 12)),  # Indices 9-17 for right arm
            "LA": list(range(12, 18)),  # Indices 18-26 for left arm
            "RL": list(range(18, 24)),  # Indices 27-35 for right leg
            "LL": list(range(24, 30)),  # Indices 36-44 for left leg
            "NSensors":5
            # The final column "activityID" is at index 45 but is not part of the sensor data.
        },
        "HAR70PLUS": {
            # HAR70PLUS has two sensor locations.
            "back": list(range(0, 3)),  # back_x, back_y, back_z → indices 0-2
            "thigh": list(range(3, 6)),  # thigh_x, thigh_y, thigh_z → indices 3-5
            "nSensor": 5
            # "activityID" is at index 6.
        },
        "HARTH": {
            # HARTH is similar to HAR70PLUS, but the label is called "label".
            "back": list(range(0, 3)),  # indices 0-2
            "thigh": list(range(3, 6)),  # indices 3-5
            # "label" is at index 6.
        },
        "MHEALTH": {
            # MHEALTH has three sensors, with some sensors giving multiple modalities.
            "left_ankle_sensor": list(range(0, 6)),  # acceleration (3-5) and gyro (6-8)
            "right_lower_arm_sensor": list(range(6, 12)),  # acceleration (9-11) and gyro (12-14)
            "NSensors": 2
            # "activityID" is at index 15.
        },
        "MOTIONSENSE": {
            # MOTIONSENSE is a single IMU with 3 acceleration + 3 gyro readings.
            "imu": list(range(0, 6)),  # indices 0-5
        },
        "OPPORTUNITY": {
            "BACK": list(range(0, 6)),  # Columns 0-5: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
            "RUA": list(range(6, 12)),  # Columns 6-11
            "RLA": list(range(12, 18)),  # Columns 12-17
            "LUA": list(range(18, 24)),  # Columns 18-23
            "LLA": list(range(24, 30))  # Columns 24-29
        },
        "PAMAP2": {
                   "BACK": list(range(0, 6)),  # Columns 0-5: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
                    "RUA": list(range(6, 12)),  # Columns 6-11
                    "RLA": list(range(12, 18)),  # Columns 12-17
                    "NSensors": 3
                    },
        "REALDISP": {
                    "1": list(range(0, 6)),  # Columns 0-5: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
                    "2": list(range(6, 12)),  # Columns 6-11
                    "3": list(range(12, 18)),  # Columns 12-17
                    "4": list(range(18, 24)),  # Columns 18-23
                    "5": list(range(24, 30)),  # Columns 24-29
                    "6":list(range(30, 36)),
                    "7":list(range(36, 42)),
                    "8":list(range(42, 48)),
                    "9":list(range(48, 54)),
                    "NSensors": 9
                    }
    }
    return datasets[dataset]



def get_data_single_sensor(dataset_instance, indices):
    """
    Calls the necessary methods on the dataset instance to generate training, validation, and test splits.
    """
    if dataset_instance is None:
        raise ValueError("Empty dataset instance")
    else:
        dataset_instance.get_datasets()
        dataset_instance.preprocessing()
        dataset_instance.normalize()
        dataset_instance.data_segmentation()
        dataset_instance.prepare_dataset()
        print(indices)

        print(dataset_instance.training_final[0][0].shape)

        # Expecting that these attributes are set by prepare_dataset()
        train = [(a[0][indices,:], a[1], a[2]) for a in dataset_instance.training_final]
        validation = [(a[0][indices,:], a[1], a[2]) for a in dataset_instance.validation_final]
        test = [(a[0][indices,:], a[1], a[2]) for a in dataset_instance.testing_final]

        print(train[0][0].shape)


        fancy_print(f"The length of the training data is {len(train)}")
        fancy_print(f"The length of the validation data is {len(validation)}")
        fancy_print(f"The length of the testing data is {len(test)}")

    return {"Train": train, "Validation": validation, "Test": test}



def get_dataloader_single_sensor(args):
    LOSO_DISTRIBUTION = None
    set_random_seed(args.seed)

    try:
        # Load LOSO distributions (if needed for future use)
        with open("./LOSO_DISTRIBUTIONS.yaml", "r") as f:
            LOSO_DISTRIBUTION = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] We failed loading the distribution file. Check if it exists")

    LOSO_DISTRIBUTION = LOSO_DISTRIBUTION.get(args.dataset)[f'{args.experiment}']
    

    dataset_classes = import_dataset_classes("dataset")
    dataset_instance = dataset_classes[args.dataset](
        train=LOSO_DISTRIBUTION.get('train'),
        validation=LOSO_DISTRIBUTION.get('validation'),
        test=LOSO_DISTRIBUTION.get('test'),
        current_directory=DATA_PATH
    )
    print("\nDataset instances created:")
    print(dataset_instance)
    SensorIndecesDictionary = _get_sensors_indices(args.dataset)
    KeysDatasetInformation = list(SensorIndecesDictionary.keys())
    sensor_index = SensorIndecesDictionary[KeysDatasetInformation[args.sensorPosition]]


    data_distributions = get_data_single_sensor(dataset_instance, sensor_index)
    DATALOADERS = {
        'Train': DataLoader(data_distributions['Train'], batch_size=args.bs, shuffle=True, drop_last=True),
        'Validation': DataLoader(data_distributions['Validation'], shuffle = True, batch_size=32),
        'Test': DataLoader(data_distributions['Test'], batch_size=args.bs)
    }

    return DATALOADERS


def group_all_data_by_person(dataset_instance):
    """
    Combine all samples (Train, Validation, Test) from `dataset_instance`
    into a single list, then group them by person_label in a dictionary.

    Returns:
        person_dict (dict): { person_label: [ (signals, activity_label, person_label), ...], ... }
    """
    # 1) Gather all samples from train, validation, test into one list
    all_samples = []
    for sample in dataset_instance['Train']:
        all_samples.append(sample)
    for sample in dataset_instance['Validation']:
        all_samples.append(sample)
    for sample in dataset_instance['Test']:
        all_samples.append(sample)
    
    # 2) Create a dictionary keyed by person_label
    person_dict = {}
    for sample in all_samples:
        # Typically sample might be (signals, activity, person)
        signals, activity, person = sample

        if person not in person_dict:
            person_dict[person] = []
        person_dict[person].append(sample)

    return person_dict



def get_dataloader_LOSO_Evaluations(args):
    LOSO_DISTRIBUTION = None

    try:
        # Load LOSO distributions (if needed for future use)
        with open("./LOSO_DISTRIBUTIONS.yaml", "r") as f:
            LOSO_DISTRIBUTION = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] We failed loading the distribution file. Check if it exists")

    LOSO_DISTRIBUTION = LOSO_DISTRIBUTION.get(args.dataset)[f'{args.experiment}']

    dataset_classes = import_dataset_classes("dataset")
    dataset_instance = dataset_classes[args.dataset](
        train=LOSO_DISTRIBUTION.get('train'),
        validation=LOSO_DISTRIBUTION.get('validation'),
        test=LOSO_DISTRIBUTION.get('test'),
        current_directory=DATA_PATH
    )
    data_distributions = get_data(dataset_instance)

    print("\nDataset instances created:")
    print(dataset_instance)
    data_distributions = group_all_data_by_person(data_distributions)

    DATALOADERS = {}


    for key, values in data_distributions.items():
        print(len(values))
        DATALOADERS[key] = DataLoader(values, batch_size=args.bs)

    return DATALOADERS



# ╔═════════════════════════════════╗
# ║ Classes                         ║
# ╚═════════════════════════════════╝




class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""
    def __init__(self, patience=7,delta=0, verbose=True , path='checkpoint.pt', trace_func=print):
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
        self.early_stop = {}
        self.val_acc_max = float('-inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.MODELS_COPY={}

    def save_model(self, model, name, paths):
        if not os.path.exists(paths):
            # Create the directory
            os.makedirs(paths)
            print(f'Directory {paths} created')
        else:
            print(f'Directory {paths} already exists')
        torch.save(model, name)

    def __call__(self, val_acc, model,keys,PATH, args):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            path_models = DATA_PATH + PATH + f'tmp/'
            os.makedirs(path_models, exist_ok=True)  # Ensure the directory exists
            print(f"Saving models in {path_models} for early stopping")
            save_path = os.path.join(path_models, f"{keys}.pt")
            print(f"Saving {keys} to {save_path}")
            self.save_model(model, save_path, path_models)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter == self.patience:
                self.early_stop = True
                self.val_acc_max = self.best_score
            elif self.counter > self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            path_models = DATA_PATH + PATH + f'tmp/'
            os.makedirs(path_models, exist_ok=True)  # Ensure the directory exists
            print(f"Saving models in {path_models} for early stopping")
            save_path = os.path.join(path_models, f"{keys}.pt")
            print(f"Saving {keys} to {save_path}")
            self.save_model(model, save_path, path_models)


    def save_checkpoint(self, val_acc,keys, PATH, args):
        """Saves model when validation accuracy increase."""
        path_models_FINAL = DATA_PATH + PATH
        os.makedirs(path_models_FINAL, exist_ok=True)  # Ensure the directory exists
        if self.verbose:
            self.trace_func(f'Validation accuracy increased ({self.val_acc_max:.6f}% --> {val_acc:.6f}%). Saving model ...')
        path_models_TMP = DATA_PATH + PATH + 'tmp/'
        save_path_TMP = os.path.join(path_models_TMP, f"{keys}.pt")
        model = torch.load(save_path_TMP)
        self.save_model(model,keys,path_models_FINAL)
        return model

                
        
        # torch.save(model.state_dict(), self.path)
        # self.val_acc_max = val_acc


import torch
from torch import nn
from pytorch_metric_learning.miners import BaseMiner
import torch.nn.functional as F

class SensorActivityMiner(BaseMiner):
    """
    A custom miner that:
      - Defines positives as (same sensor, same activity).
      - Defines hard negatives as (same sensor, different activity).
      - Defines weak negatives as (different sensor).
    
    We'll return triplets (a, p, n).
    We combine 'hard' and 'weak' negatives into the final negative set.
    You can adapt the "hard negative" logic to pick only the 'closest' 
    same-sensor-diff-activity sample if you want.

    If you want to do "batch-hard" style, we'll pick for each anchor:
      - The hardest positive (furthest in distance)
      - The hardest negative (closest in distance) among both 'hard' and 'weak' sets
        or keep them separate if you prefer.

    By default, BaseTupleMiner expects:
      mine(embeddings, labels, ref_emb=None, ref_labels=None)
      returns (anchors, positives, negatives) as index tensors.
    """

    def __init__(self, distance=None, margin_hard=0.2, margin_weak=0.2, 
                 separate_negatives=False, **kwargs):
        super().__init__(distance=distance, **kwargs)
        self.margin_hard = margin_hard
        self.margin_weak = margin_weak
        self.separate_negatives = separate_negatives
        # distance is typically a Distance object from pytorch-metric-learning 
        # (e.g., CosineSimilarity or LpDistance). If None, defaults to L2 distance.

    def mine(self, embeddings, labels, ref_emb=None, ref_labels=None):
        """
        embeddings: shape (batch_size, embed_dim)
        labels: shape (batch_size,) - must contain or encode sensor & activity

        We can either:
          1) store sensor_id, activity_id in separate arrays in `labels`, OR
          2) pack them into a single integer label, then decode it here.
             For example: label = sensor_id * 1000 + activity_id.

        We'll assume below you have two separate arrays for clarity.
        """
        # If you literally have "labels" as a single int, you'll need a way to decode it:
        # sensor_ids = labels // 1000
        # activity_ids = labels % 1000
        # Or:
        # parse from a custom approach.

        # EXAMPLE: Suppose your code (outside) calls:
        # miner.mine(emb, (sensor_ids, activity_ids))
        # Then 'labels' is actually a tuple or something like that. We'll handle that.

        if isinstance(labels, tuple):
            sensor_ids, activity_ids = labels
        else:
            raise ValueError("labels must be a (sensor_ids, activity_ids) tuple for this custom miner.")

        # Convert to device, if needed
        sensor_ids = sensor_ids.to(embeddings.device)
        activity_ids = activity_ids.to(embeddings.device)

        batch_size = embeddings.shape[0]

        # 1) Pairwise distances or similarities
        # If self.distance is None, let's do a default L2. 
        # If self.distance is e.g. CosineSimilarity, note that you'd have 
        # a "distance" = 1 - cos_sim or something like that. We'll assume L2 for demonstration.

        if self.distance is not None:
            # pytorch-metric-learning distance object
            dist_mat = self.distance(embeddings)
            # Depending on the distance object, if it's CosineSimilarity, 
            # you might get "similarities" not "distances".
            # Check self.distance.inverse_distance for actual distances, etc.
            # We'll assume it returns a distance matrix shape (batch_size, batch_size).
        else:
            # default L2:
            # cdist(embeddings, embeddings):
            dist_mat = torch.cdist(embeddings, embeddings, p=2)

        # 2) Build masks
        same_sensor   = sensor_ids.unsqueeze(1).eq(sensor_ids.unsqueeze(0))
        same_activity = activity_ids.unsqueeze(1).eq(activity_ids.unsqueeze(0))

        pos_mask  = same_sensor & same_activity  # same sensor + same activity
        hard_mask = same_sensor & (~same_activity) # same sensor, diff activity
        weak_mask = ~same_sensor                # different sensor

        # We also don't want i=j, so exclude the diagonal for positives/negatives
        diag_idx = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        pos_mask[diag_idx] = False
        hard_mask[diag_idx] = False
        weak_mask[diag_idx] = False

        # 3) For each anchor i, find one positive, one negative (or multiple).
        # We'll do a "batch-hard" approach: hardest positive = the furthest positive,
        # hardest negative = the closest negative among both hard & weak sets. 
        # But you can do separate "hard negative" vs. "weak negative" if you prefer.

        # dist_ap[i] = distances to positives for anchor i
        # where not positive => 0
        zero_mat = torch.zeros_like(dist_mat)
        inf_mat  = torch.full_like(dist_mat, float('inf'))

        dist_ap = torch.where(pos_mask, dist_mat, zero_mat)
        # hardest positive = max distance
        hardest_pos_val, hardest_pos_idx = dist_ap.max(dim=1)

        # We'll define a single negative set = union of hard_mask, weak_mask
        neg_mask = hard_mask | weak_mask
        dist_an  = torch.where(neg_mask, dist_mat, inf_mat)
        # hardest negative = min distance
        hardest_neg_val, hardest_neg_idx = dist_an.min(dim=1)

        # Now we have, for each anchor i:
        #   p = hardest_pos_idx[i]
        #   n = hardest_neg_idx[i]
        # But some anchors might have NO positives (hardest_pos_val = 0) 
        # if none in the batch share sensor+activity. We should exclude them.

        anchors = torch.arange(batch_size, device=embeddings.device)
        positives = hardest_pos_idx
        negatives = hardest_neg_idx

        # Filter out anchors that have no valid positive:
        # "hardest_pos_val = 0" => means there's no pos in that row
        valid_mask = (hardest_pos_val > 0)
        anchors   = anchors[valid_mask]
        positives = positives[valid_mask]
        negatives = negatives[valid_mask]

        # Return the triplets in the format (anchors, positives, negatives).
        return anchors, positives, negatives