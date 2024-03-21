import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from load_audio_mnist import load_audioMNIST


class AudioMNIST(Dataset):

    def __init__(self, path, sample_rate, length):
        super().__init__()
        self.X, self.y = load_audioMNIST(path, sample_rate, length)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


class AudioModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Conv BLock 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), bias=False, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3), bias=False, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), bias=False, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3456, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.linear(x)

        return x


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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")


    # =========================== DATA LOADING ===========================
    DATA_PATH = "data/AudioMNIST/"
    SAMPLE_RATE = 8000
    LENGTH = 1000

    dataset = AudioMNIST(DATA_PATH, SAMPLE_RATE, LENGTH)

    split = 0.8
    train_len = int(len(dataset) * split)
    test_len = len(dataset) - train_len

    train_ds, test_ds = random_split(dataset, [train_len, test_len])

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)


    # =========================== MODEL TRAINING ===========================
    # model = AudioModel()
    # model = model.to(device)

    # fit(model, train_dl, 20)

    # # Saving Model (With TorchScript)
    # model = torch.jit.script(model)
    # torch.jit.save(model, "models/test_model_02.pth")


    # =========================== EVALUATION ===========================
    model = torch.jit.load("models/test_model_02.pth")
    model.eval()    # Enable inference mode
    eval(model, test_dl)
