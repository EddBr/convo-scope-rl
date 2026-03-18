from reward.Base_Reward import Base_Reward
from agent.Conversation import Conversation
import numpy as np

# Reward function that returns the length of the human response
class Human_Length_Reward(Base_Reward):
    def get_reward(self, prev_state : Conversation, action : str, human_response : str) -> float:
        if human_response is None:
            #End of convo
            return 0
        if isinstance(human_response,str):
            #eval
            # I can't embed - requires booting up model
            return 0
        dot_prod = np.dot(action, human_response)
        norm_action = np.linalg.norm(action)
        norm_human = np.linalg.norm(human_response)
        similarity = dot_prod / (norm_action * norm_human)
        dist = 1 - similarity
        return - dist # Negative because the paper found it to be bad
        #return 0.01*len(human_response)
