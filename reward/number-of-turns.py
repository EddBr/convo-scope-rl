from reward.Base_Reward import Base_Reward
import torch
import torch.nn as nn
from token_count import TokenCount

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

class Embedding_Length_Reward(Base_Reward):
    
    def __init__(self, add_llm_length : bool, path_to_model="reward/embedding_length_reward", device_map=0) -> None:
        super().__init__()
        print(f"Loading embedding length model on device {device_map}...")
        self.model = MLPRegression()
        #self.model.load_state_dict(torch.load(path_to_model, map_location=torch.device(device_map)))
        self.add_llm_length = add_llm_length
        print("length model initialized with add_llm_length: ", self.add_llm_length)

    def get_reward(self, prev_state : Conversation | tuple, action : str | tuple, human_response : str | tuple | None) -> float:

        if human_response is None:
            if isinstance(action, str): #if string
                score_after = self.get_safe_prob((prev_state + action).create_chat())
                score_before =  self.get_safe_prob(prev_state.create_chat())
                reward = (score_after - score_before)
                # print("before multipling reward is ", reward)
                return 1000 * reward
            else: # if semantic
                reward = self.get_safe_prob_from_embedding(torch.FloatTensor(prev_state) + torch.FloatTensor(action)) - self.get_safe_prob_from_embedding(torch.FloatTensor((prev_state)))
                # print("before multipling reward is ", reward)
                return 1000 * reward

        if isinstance(prev_state, Conversation):
            score_after = self.get_safe_prob((prev_state + action + human_response).create_chat())
            score_before =  self.get_safe_prob(prev_state.create_chat())
            reward = (score_after - score_before)
            # print("before multipling reward is ", reward)
            return 1000 * reward
        else:
            reward = self.get_safe_prob_from_embedding(torch.FloatTensor((human_response))) - self.get_safe_prob_from_embedding(torch.FloatTensor((prev_state)))
            # print("before multipling reward is ", reward)
            return 1000 * reward
