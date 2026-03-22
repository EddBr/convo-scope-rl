from reward.Base_Reward import Base_Reward
import numpy as np
import torch
import torch.nn as nn
from agent.Conversation import Conversation


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

class LLM_Judge_Reward(Base_Reward):
    def __init__(self, path_to_model="reward/judge_model.pth", device_map=0) -> None:
        super().__init__()
        print(f"Loading embedding length model on device {device_map}...")
        self.model = MLPRegression()
        self.model.load_state_dict(torch.load(path_to_model, map_location=torch.device(device_map)))
    def get_reward(self, prev_state : Conversation, action : str, human_response : str) -> float:
        try:
            return self.model(torch.FloatTensor(action))
        except:
            return 3.5

