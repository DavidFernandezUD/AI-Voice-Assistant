"""
Utility script for generating a csv file for wakeword dataset.
"""


import csv
import os
import argparse


def create_csv(dataset_path: str, csv_path: str, multiplier: int = 1):
    
    path_0 = os.path.join(dataset_path, "0")
    path_1 = os.path.join(dataset_path, "1")

    data = list()
    for file in os.listdir(path_0):
        
        file_path = os.path.join(path_0, file)
        data.append({"audio": file_path, "label": 0})

    for _ in range(multiplier):
        for file in os.listdir(path_1):

            file_path = os.path.join(path_1, file)
            data.append({"audio": file_path, "label": 1})

    with open(csv_path, "w") as file:
        
        writer = csv.DictWriter(file, fieldnames=["audio", "label"])
        writer.writerows(data)  # NOTE: Not writting header


def create_csv2(dataset_path: str, csv_path: str, multiplier: int = 1):
    
    path_0 = "data/speech_commands_v0.02/"
    path_1 = os.path.join(dataset_path, "1")

    data = list()
    for dir in os.listdir(path_0):
        dir_path = os.path.join(path_0, dir)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)

                if dir == "marvin":
                    for _ in range(multiplier):
                        data.append({"audio": file_path, "label": 1})
                else:
                    data.append({"audio": file_path, "label": 0})

    with open(csv_path, "w") as file:
        
        writer = csv.DictWriter(file, fieldnames=["audio", "label"])
        writer.writerows(data)  # NOTE: Not writting header

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("dataset_path", type=str, default=None)
    # parser.add_argument("csv_path", type=str, default=None)
    # parser.add_argument("--multiplier", type=int, default=1, required=False)
    
    # args = parser.parse_args()

    # create_csv(args.dataset_path, args.csv_path, args.multiplier)
    create_csv2("data/wakeword_dataset/train/", "data/wakeword_dataset/train_02.csv", multiplier=25)

    # Example:
    # python create_wakeword_dataset_csv.py data/wakeword_dataset/train/ data/wakeword_dataset/train.csv --multiplier 600
