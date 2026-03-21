import torch
import torch.nn as nn
from token_count import TokenCount
import datasets
import numpy as np
from tqdm import tqdm

end = 10_000

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

dataset = datasets.load_from_disk(f"/home/s2289391/convo-scope-rl/judge/embedding_rewards/lmsys_embeddings_rewards_{end}").with_format("torch")

dataset = dataset.cast_column("reward", datasets.Value("float32"))

print("Creating test/train split")
split = dataset.train_test_split(test_size=0.2)

train_loader = torch.utils.data.DataLoader(split["train"],batch_size=2048, shuffle=True)
test_loader = torch.utils.data.DataLoader(split["test"],batch_size=2048, shuffle=True)

print("Loaded train/test split")

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
        inputs = batch["embedding"].to(device)
        #inputs = (inputs - stats["mean"]) / (stats["std"])
        targets = batch["reward"].to(device).float().unsqueeze(1)

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
            inputs = batch["embedding"].to(device)
            #inputs = (inputs - stats["mean"]) / (stats["std"])
            targets = batch["reward"].to(device).float().unsqueeze(1) 

            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
        print("val loss",str(val_loss))

torch.save(model.state_dict(), "judge_model.pth")
print("SAVED JUDGE MODEL")
