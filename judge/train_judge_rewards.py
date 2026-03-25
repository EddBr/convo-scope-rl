from datasets import load_from_disk, Dataset
from tqdm import tqdm
import torch.nn as nn

start = 0
end = 130_000


dataset = load_from_disk("judgements/lmsys_130000_final")

class MLPRegression(nn.Module):
    def __init__(self):
        super(MLPRegression, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = (self.fc5(x))
        return x


def create_reward_data(batch):
    conversations = batch["embeddings"]
    inputs = []
    targets = []
    for convo in conversations:
        n = len(convo)
        for c in range(1,n,2):
            inputs.append(convo[c])
            targets.append(n - c - 1)

    return {
            "inputs":inputs,
            "targets":targets
            }

dataset = dataset.select(end)



# Batching is more efficient. Lemme do it stupid first
reward_ds = dataset.map(
        create_reward_data,
        batched=True,
        remove_columns=embeddings.column_names,
        num_proc=16
        )

device = "cuda"
model = MLPRegression().to(device)
optimiser = torch.optim.Adam(model.parameters(),lr=1e-4)
criterion = nn.MSELoss()

num_epochs = 100
print("Beginning Training")
for epoch in tqdm(range(0,num_epochs),leave=True):
    print("Epoch:", str(epoch))
    model.train()
    train_loss = 0.0
    for batch_no, batch in enumerate(tqdm(train_loader, leave=True)):
        inputs = batch["inputs"].to(device)
        inputs = (inputs - stats["mean"]) / (stats["std"])
        targets = batch["targets"].to(device).float().unsqueeze(1)

        optimiser.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimiser.step()

        train_loss += loss.item()
    print("train loss",str(train_loss))

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            inputs = (inputs - stats["mean"]) / (stats["std"])
            targets = batch["targets"].to(device).float().unsqueeze(1) 

            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
        print("val loss",str(val_loss))
