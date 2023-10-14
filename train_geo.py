import argparse
import torch
from datasets import load_dataset, load_from_disk
from transformers import ViTImageProcessor
from datasets import DatasetDict
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from transformers import ViTModel

import torch.optim as optim
import os


DIR = "CS674/project1"

SAVE_PATH_DATASET = f"{DIR}/dataset/"
SAVE_PATH_IMAGE_PROCESSOR = f"{DIR}/processor/"
VIT_BASE_DIR = f"{DIR}/checkpoints/"
SAVE_PATH_VIT_BASE_MODEL = f"{DIR}/base_ViT/"


VIT_BASE_FILENAME = f"{VIT_BASE_DIR}/checkpoint"
EPOCH_FILE = f"{DIR}/epoch.txt"
LOSSES_FILE = f"{DIR}/losses.txt"
VALIDATION_LOSSES_FILE = f"{DIR}/validation_losses.txt"


HUGGING_FACE_PRETRAINED_VIT = "google/vit-base-patch16-224-in21k"
HUGGING_FACE_PRETRAINED_VIT_PROCESSOR = "google/vit-base-patch16-224"
HUGGING_FACE_DATASET_KEY = "babananabananana/long_lat_maps"


class CustomViTRegressor(nn.Module):
    def __init__(self, should_load_from_disk=True, base_filename=VIT_BASE_FILENAME, base_dir=VIT_BASE_DIR):
        super(CustomViTRegressor, self).__init__()
        if should_load_from_disk:
            self.base_model = ViTModel.from_pretrained(SAVE_PATH_VIT_BASE_MODEL)
        else:
            self.base_model = ViTModel.from_pretrained(HUGGING_FACE_PRETRAINED_VIT)
            self.base_model.save_pretrained(SAVE_PATH_VIT_BASE_MODEL)
        self.regressor = nn.Linear(self.base_model.config.hidden_size, 2)  # 2 for longitude and latitude
        self.base_filename = base_filename
        self.base_dir = base_dir

    def forward(self, x):
        features = self.base_model(x)
        predictions = self.regressor(features.pooler_output)
        return predictions

    def saved_model_exists(self):
        if not os.path.exists(self.base_filename):
            return False
        return True

    def update_model_from_checkpoint(self):
        if not self.saved_model_exists():
            return "no saved model exists"

        path = self.get_last_file()

        if not os.path.exists(path):
            raise Exception(f"Model does not exist! {path}")
        loaded = torch.load(path)
        return self.load_state_dict(loaded)

    def get_last_file(self):
        checkpoint_paths = os.listdir(self.base_dir)
        trimmed_paths = []
        for path in checkpoint_paths:
            trimmed_paths.append(path[16:])
        if '' in trimmed_paths:
            trimmed_paths.remove('')
        if len(trimmed_paths) == 0:
            return self.base_filename

        epochs = []
        for epoch in trimmed_paths:
            epochs.append(int(epoch))
        epochs.sort()
        largest_checkpoint = epochs[-1]
        return f"{self.base_filename}_epoch{largest_checkpoint}"

    def get_unique_filename(self, epoch):
        if not os.path.exists(self.base_filename):
            return self.base_filename
        #               i.e. "save/path/checkpoint_epoch5
        starting_filename = f"{self.base_filename}_epoch{epoch}"

        if os.path.exists(starting_filename):
            raise Exception("Already saved file with that epoch")
        return starting_filename

    def save(self, epoch):
        output_filename = self.get_unique_filename(epoch)
        torch.save(self.state_dict(), output_filename)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--access_internet",
        type=int,
        default=0,
        help="default=0 (don't access) If we should access the internet.  "
             "If provided, gets models and datasets from the internet and saves them to the disk.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def train_test_valid_split(dataset, valid_percent, test_percent, seed=42):
    # Split once into train and (valid and test)

    train_test_valid = dataset.train_test_split(test_size=valid_percent + test_percent, seed=seed)

    # Split (valid and test) into train and test - call the train of this part validation
    new_test_size = test_percent / (valid_percent + test_percent)
    test_valid = train_test_valid['test'].train_test_split(test_size=new_test_size, seed=seed)

    # gather all the pieces to have a single DatasetDict
    train_test_valid_dataset = DatasetDict({
        'train': train_test_valid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test'],
        })
    return train_test_valid_dataset


def custom_data_collator_function(processor):
    def return_func(batch):
        images = [item["image"].convert("RGB") for item in batch]
        inputs = processor(images, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        targets = torch.tensor(
            [[item["latitude"], item["longitude"]] for item in batch],
            dtype=torch.float32)

        return inputs, targets

    return return_func


def euclidean_distance_loss(y_prediction, y_true):
    # Calculate Euclidean distance
    distance = torch.sqrt(torch.sum((y_prediction - y_true)**2, dim=1))
    # Take the mean distance as the loss
    loss = torch.mean(distance)
    return loss


def _train(model,
           num_epochs_to_train,
           loss_values,
           validation_loss_values,
           train_loader,
           val_loader,
           optimizer,
           objective,
           checkpoint_model=True,
           save_frequency=1,
           device='cuda'):

    for epoch in range(num_epochs_to_train):
        # GETTING EPOCH NUMBER
        with open(f"{EPOCH_FILE}", "r") as f:
            for line in f:
                total_epochs = int(line.strip())
                break

        # CHECKING VALIDATION LOSS BEFORE TRAINING
        vl = []
        for x_v, y_v in val_loader:
            x_v, y_v = x_v['pixel_values'].to(device), y_v.to(device)
            vl.append(objective(model(x_v), y_v).item())
            val = np.mean(vl)
            validation_loss_values.append((total_epochs, val))

        # DO AN EPOCH
        batch_losses = []
        for batch, (x, y_truth) in enumerate(train_loader):  # learn
            x, y_truth = x['pixel_values'].to(device), y_truth.to(device)

            optimizer.zero_grad()
            y_hat = model(x)

            loss = objective(y_hat, y_truth)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()

        # EPOCH OVER, INCREMENT EPOCHS AND SAVE NEW VALUE
        total_epochs += 1
        with open(f"{EPOCH_FILE}", "w") as f:
            f.write(str(total_epochs) + "\n")

        # UPDATE TRAINING SET LOSSES
        loss_values.append((total_epochs, np.mean(batch_losses)))

        # CHECKPOINT MODEL
        if (checkpoint_model and total_epochs % save_frequency == 0) or (checkpoint_model and total_epochs == 1):
            model.save(total_epochs)

            with open(f"{LOSSES_FILE}", "w") as f:
                for loss in loss_values:
                    f.write(str(loss) + "\n")

            with open(f"{VALIDATION_LOSSES_FILE}", "w") as f:
                for val_loss in validation_loss_values:
                    f.write(str(val_loss) + "\n")


def main(args):
    if args.access_internet == 1:
        dataset = load_dataset(HUGGING_FACE_DATASET_KEY)
        dataset = train_test_valid_split(dataset['train'], .15, .15)
        dataset.save_to_disk(SAVE_PATH_DATASET)

        image_processor = ViTImageProcessor.from_pretrained(HUGGING_FACE_PRETRAINED_VIT_PROCESSOR)
        image_processor.save_pretrained(SAVE_PATH_IMAGE_PROCESSOR)
        device = "cuda"         # TODO CHANGE THIS BEFORE SENDING TO DOCKER. NEEDS TO BE "CPU" BECAUSE THE LOGIN NODE DOESN'T HAVE CUDA
    else:
        device = "cuda"
        dataset = load_from_disk(SAVE_PATH_DATASET)
        image_processor = ViTImageProcessor.from_pretrained(SAVE_PATH_IMAGE_PROCESSOR)

    custom_data_collator = custom_data_collator_function(image_processor)

    train_loader = DataLoader(dataset["train"], batch_size=32, collate_fn=custom_data_collator, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=32, collate_fn=custom_data_collator)
    # test_loader = DataLoader(dataset["test"], batch_size=32, collate_fn=custom_data_collator)

    if args.access_internet == 1:
        model = CustomViTRegressor(should_load_from_disk=False).to(device)
        losses = []
        val_losses = []
    else:
        model = CustomViTRegressor().to(device)
        success = model.update_model_from_checkpoint()
        print(success)
        losses = []
        val_losses = []
        with open(f"{LOSSES_FILE}", "r") as f:
            for line in f:
                line = line.strip()
                line = line[1:-1]
                nums = line.split(',')
                epoch = int(nums[0])
                loss = float(nums[1])

                losses.append((epoch, loss))

        with open(f"{VALIDATION_LOSSES_FILE}", "r") as f:
            for line in f:
                line = line.strip()
                line = line[1:-1]
                nums = line.split(',')
                epoch = int(nums[0])
                loss = float(nums[1])

                val_losses.append((epoch, loss))

    objective = euclidean_distance_loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    _train(model, args.max_train_steps, losses, val_losses, train_loader, val_loader, optimizer, objective, device=device)


if __name__ == "__main__":
    main(parse_args())
