import os
from tqdm import tqdm
import numpy as np

from data import Data
from model import Model

import torch


class Eval:
    def __init__(self, data_path, weights_path):

        self.data_handler = Data(data_path)
        _, _, self.test_loader = self.data_handler.get_loaders()

        self.weights_dir = os.path.expanduser(weights_path)

        # Initialise model for inference
        self.model = Model()
        self.model.load_state_dict(torch.load(self._get_latest_weights()))
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _get_latest_weights(self):
        """
        Given the path to the weights directory, and assuming that file
        names are of format: alexnet_epoch_X_pth
        Get the most recent weights file path
        """

        files = os.listdir(self.weights_dir)
        pth_files = [file for file in files if file.endswith('.pth')]
        if not pth_files:
            return None  # No matching files found

        sorted_files = sorted(pth_files, key=lambda x: int(
            x.split('_epoch_')[1].split('.')[0]))
        greatest_file = sorted_files[-1]
        return os.path.join(self.weights_dir, greatest_file)

    def model_accuracy(self):
        test_iter = iter(self.test_loader)
        sample_inputs, sample_labels = next(test_iter)
        self.get_accuracy_for_batch(sample_inputs, sample_labels)

        batch_accuracies = []

        for i, (sample_inputs, sample_labels) in tqdm(enumerate(self.test_loader)):
            batch_accuracies.append(self.get_accuracy_for_batch(
                sample_inputs, sample_labels))

        model_accuracy = sum(batch_accuracies) / len(batch_accuracies)

        print(f"Model is {model_accuracy:.2f}% accurate.")

    def get_accuracy_for_batch(self, sample_inputs, sample_labels) -> float:
        """
        Given a batch from the train split, calculate the inference accuracy
        """

        sample_inputs, sample_labels = sample_inputs.to(
            self.device), sample_labels.to(self.device)

        with torch.no_grad():
            output = self.model(sample_inputs)

        # Convert the output to probabilities using softmax
        probabilities = torch.nn.functional.softmax(output, dim=1)

        predicted_classes = torch.argmax(probabilities, dim=1)

        predicted_classes = predicted_classes.cpu().numpy()
        sample_labels = sample_labels.cpu().numpy()

        # Find the common elements and calculate accuracy
        common_elements = np.equal(predicted_classes, sample_labels)
        accuracy = np.sum(common_elements) / len(predicted_classes) * 100
        return accuracy


def main():

    project_root = "~/code/personal/alexnet"
    data_path = project_root + "/data"
    weights_path = project_root + "/weights"

    eval = Eval(data_path=data_path,
                weights_path=weights_path)

    eval.model_accuracy()


if __name__ == "__main__":
    main()
