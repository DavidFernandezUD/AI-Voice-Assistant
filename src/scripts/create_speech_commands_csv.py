"""
Utility script for generating a json file for speech_commands dataset.
"""


import csv
import os
import argparse


CLASSES = [
    'no', 'learn', 'bed', 'marvin', 'zero',
    'six', 'yes', 'eight', 'up', 'on',
    'visual', 'sheila', 'wow', 'stop',
    'seven', 'house', 'nine', 'forward',
    'cat', 'follow', 'right', 'bird',
    'down', 'backward', 'four', 'off',
    'one', 'happy', 'go', 'two', 'dog',
    'five', 'three', 'left', 'tree'   
]


def create_csv(dataset_path: str, csv_path: str):
    
    data = list()
    for dir in os.listdir(dataset_path):
        dir_path = os.path.join(dataset_path, dir)
        if os.path.isdir(dir_path) and dir in CLASSES:
            for file in os.listdir(dir_path):
                
                file_path = os.path.join(dir_path, file)
                label = CLASSES.index(dir)

                data.append({"audio": file_path, "label": label})

    with open(csv_path, "w") as file:
        
        writer = csv.DictWriter(file, fieldnames=["audio", "label"])
        writer.writerows(data)  # NOTE: Not writting header


def create_noise_csv(dataset_path: str, csv_path: str):

    assert "_background_noise_" in os.listdir(dataset_path)

    data = list()

    dir_path = os.path.join(dataset_path, "_background_noise_")
    for file in os.listdir(dir_path):
        data.append(os.path.join(dir_path, file))

    with open(csv_path, "w") as file:
        for row in data:
            file.write(row + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=None)
    
    args = parser.parse_args()
    DATASET_PATH = args.path

    create_csv(DATASET_PATH, os.path.join(DATASET_PATH, "dataset.csv"))
    create_noise_csv(DATASET_PATH, os.path.join(DATASET_PATH, "background_noise.csv"))

    # Example:
    # python create_speech_commands_csv.py data/speech_commands_v0.02/
