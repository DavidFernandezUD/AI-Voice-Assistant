import torch
import torch.nn as nn
from model import CommandModel
from dataset import SpeechCommandsDataset
from torch.utils.data import Dataset, DataLoader, random_split


DATA_PATH = "data/AudioMNIST/"
SAMPLE_RATE = 16000
LENGTH = 2000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")

dataset = SpeechCommandsDataset(DATA_PATH, SAMPLE_RATE)

split = 0.8
train_len = int(len(dataset) * split)
test_len = len(dataset) - train_len

train_ds, test_ds = random_split(dataset, [train_len, test_len])

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)


model = CommandModel()
model = model.to(device)


def fit(model, train_dl, epochs):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
        for i, batch in enumerate(train_dl):
            X_batch = batch[0].to(device)
            y_batch = batch[1].to(device)

            # Normalize
            X_batch = (X_batch - X_batch.mean()) / X_batch.std()

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


def eval(model, test_dl):

    with torch.no_grad():
        accuracy = 0.0
        n_iters = len(test_dl)
        for batch in test_dl:
            X_batch = batch[0].to(device)
            y_batch = batch[1].to(device)

            X_batch = (X_batch - X_batch.mean()) / X_batch.std()

            pred = model(X_batch)

            pred_class = torch.argmax(pred, dim=1)
            accuracy += (pred_class == y_batch).sum().item() / (n_iters * pred.shape[0])

    print(f"accuracy -> {accuracy:.2f}")


if __name__ == "__main__":

    # Train
    # fit(model, train_dl, epochs=4)
    # model = torch.jit.script(model)
    # torch.jit.save(model, "models/command_model_01.pth")

    # Evaluate
    model = torch.jit.load("models/command_model_01.pth")
    model.eval()
    eval(model, test_dl)

    
