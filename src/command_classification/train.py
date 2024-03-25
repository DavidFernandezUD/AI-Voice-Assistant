"""
Train command classification model.
"""


import torch
import torch.nn as nn
import multiprocessing
from model import CommandModel
from dataset import SpeechCommandsDataset
from torch.utils.data import DataLoader, random_split


def train(model: nn.Module, train_dl: DataLoader, epochs: int = 10, learning_rate: float = 0.001, device: str = "cpu"):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=len(train_dl),
        epochs=epochs,
        anneal_strategy="linear"
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_iters = len(train_dl)
        for batch in train_dl:

            X_batch = batch[0].to(device)
            y_batch = batch[1].to(device)

            # Reset Gradients
            optimizer.zero_grad()

            # Forward pass
            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() / n_iters

            pred_class = torch.argmax(pred, dim=1)
            epoch_acc += (pred_class == y_batch).sum().item() / (n_iters * pred.shape[0])

        print(f"Epoch {epoch + 1:>2}: loss -> {epoch_loss:.4f}, accuracy -> {epoch_acc:.2f}")


def eval(model: nn.Module, test_dl: DataLoader, device: str = "cpu"):

    with torch.no_grad():
        accuracy = 0.0
        n_iters = len(test_dl)
        for batch in test_dl:

            X_batch = batch[0].to(device)
            y_batch = batch[1].to(device)

            pred = model(X_batch)

            pred_class = torch.argmax(pred, dim=1)
            accuracy += (pred_class == y_batch).sum().item() / (n_iters * pred.shape[0])

    print(f"accuracy -> {accuracy:.2f}")


def main(
    model_paht: str,
    dataset_csv: str,
    noise_csv: str = None,
    orig_sample_rate: int = 16000,
    sample_rate: int = 16000,
    length_ms: int = 2000,
    train_split: float = 0.8,
    epochs: int = 10,
    learning_rate: int = 0.001,
    batch_size: int = 32,
    device: str = "cpu"
    ):


    # Loading dataset
    dataset = SpeechCommandsDataset(dataset_csv, noise_csv, orig_sample_rate=orig_sample_rate, sample_rate=sample_rate, length_ms=length_ms)


    # Test train split
    train_len = int(len(dataset) * train_split)
    test_len = len(dataset) - train_len

    train_ds, test_ds = random_split(dataset, [train_len, test_len])

    num_cpus = multiprocessing.cpu_count()
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_cpus)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_cpus)


    # Training
    model = CommandModel()
    model = model.to(device)
    
    train(model, train_dl, epochs=epochs, learning_rate=learning_rate, device=device)

    model.eval()
    eval(model, test_dl, device=device)


    # Compile and save model with torchscript
    model.to("cpu")
    model = torch.jit.script(model)
    torch.jit.save(model, model_paht)


if __name__ == "__main__":

    torch.manual_seed(1234)

    DATASET_CSV = "data/speech_commands_v0.02/dataset.csv"
    NOISE_CSV = "data/speech_commands_v0.02/background_noise.csv"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")

    main(
        model_paht="models/command_model_16KH.pth",
        dataset_csv=DATASET_CSV,
        noise_csv=NOISE_CSV,
        orig_sample_rate=16000,
        sample_rate=16000,
        length_ms=2000,
        train_split=0.8,
        epochs=40,
        learning_rate=0.001,
        batch_size=32,
        device=device
    )

    # ==== Base ====
    # Train: acc -> 0.34
    # Eval: acc -> 0.40

    # ==== Extra Conv + Dropout 0.25 ====
    # -- 15 epoch --
    # Train: acc -> 0.88
    # Eval: acc -> 0.91

    # ==== Extra Conv + 0.5 Dropout ====
    # -- 15 epoch --
    # Train: acc -> 0.83
    # Eval: acc -> 0.9

    # ==== Extra Conv + Dropout 0.25 + No Amp to db ====
    # -- 15 epoch --
    # Train: acc -> 0.82
    # Eval: acc -> 0.79

    # ==== Extra Conv + Dropout 0.25 + Sample Rate 8kH ====
    # -- 15 epoch --
    # Train: acc -> 0.79
    # Eval: acc -> 0.84

    # ==== Extra Conv + Dropout 0.25 + Noise rate 0.1 ====
    # -- 15 epoch --
    # Train: acc -> 0.86
    # Eval: acc -> 0.9

    # ==== Extra Conv + Dropout 0.25 + No SpecAugment ====
    # -- 15 epoch --
    # Train: acc -> 0.92
    # Eval: acc -> 0.94

    # ==== Extra Conv + Dropout 0.25 + 128 Mels ====
    # -- 15 epoch --
    # Train: acc -> 0.87
    # Eval: acc -> 0.90

    # ==== Extra Conv + Dropout 0.25 + MFCC 40  + No Amp to db ====
    # -- 15 epoch --
    # Train: acc -> 0.81
    # Eval: acc -> 0.86

    # ===================================================
    # 16kH Dropout 0.25 + SpecAugment
    # Train: -> 0.90
    # Eval: -> 0.95

    # 16kH Dropout 0.25 + No SpecAugment
    # Train: -> 0.94
    # Eval: -> 0.95

    # 16kH Dropout 0.5 + SpecAugment
    # Train: -> 0.88
    # Eval: -> 0.95

    # 8kH Dropout 0.5 + SpecAugment
    # Train: -> 0.86
    # Eval: -> 0.93
