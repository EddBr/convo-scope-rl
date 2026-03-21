from reward.Base_Reward import Base_Reward
from agent.Conversation import Conversation
import numpy as np

# Reward function that returns the length of the human response
class LLM_Judge_Reward(Base_Reward):
    def get_reward(self, prev_state : Conversation, action : str, human_response : str) -> float:
        return None
