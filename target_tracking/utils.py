import numpy as np
from icecream import ic
from stable_baselines3 import SAC
from gymnasium import spaces
import torch as th


def flatten(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten(item))
        else:
            flattened_list.append(item)
    return flattened_list

def generate_observation(env, pursuer, evader):

    stateE = evader

    maxD=2*env.max_position*np.sqrt(2)

    observation=[]
    stateP = pursuer

    # ------- evader to pursuer ----------
    pos_pur_eva = stateE[0:2] - stateP[0:2]
    dist_pur_eva = np.linalg.norm(pos_pur_eva)
    angle = np.arctan2(pos_pur_eva[1], pos_pur_eva[0])
    dist_pur_eva /= maxD 
    angle /= np.pi
    pTheta= stateP[2]/np.pi


    # ------- pursuers to ezs ----------
    dPEzs=[]
    angPEzs=[]

    for ez in env.OBSTs:
        vecPEzi = ez.pos[0:2]-stateP[0:2]
        dwez = np.clip((np.linalg.norm(vecPEzi)-ez.radius) / maxD, -1, 1)
        dPEzs.append(dwez) 
        angPEzs.append(np.arctan2(vecPEzi[1], vecPEzi[0])/np.pi) 


    dist_pur_eva = np.clip(dist_pur_eva, 0, 1)
    observation.append([dist_pur_eva, angle, pTheta, dPEzs, angPEzs])

    flattened_observation = flatten(observation)
    observation_array = np.array(flattened_observation)
    
    return observation_array


class Tactic:
    def __init__(self, env, dir_load_model, action_space_size):
        # print("Loading model: " + dir_load_model)
        action_space = spaces.Box(low=np.array(np.ones(action_space_size)*(-1), dtype=np.float32), high=np.array(np.ones(action_space_size), dtype=np.float32), dtype=np.float32)

        self.model = SAC.load(dir_load_model, custom_objects={'action_space':action_space})
        self.directory_model = dir_load_model

        self.actor = self.model.actor.forward
        self.critic = self.model.critic.forward

    def compute_V(self, obs):
        #The way it is, observation just have to be a shape
        # ic(obs)
        obs=np.array(obs,np.float64)
        # ic(obs)
        # Add a batch dimension to the observation (e.g., from shape [obs_dim] to [1, obs_dim])
        obs = np.expand_dims(obs, axis=0)
        # ic(obs)
        if th.cuda.is_available():
            obs = th.tensor(obs).cuda()
        else:
            obs = th.tensor(obs)
        # ic(obs)
        # obs=model.policy.obs_to_tensor(obs)[0]
        # ic(obs)
        act = self.actor(obs, deterministic = True)
        # ic(act)
        values=self.critic(obs, act)
        # ic(values)
        values = [value.cpu().detach().numpy() for value in values]
        # ic(obs,values,self.directory_model)

        return np.mean(values)
    
    def compute_Action(self, obs):
        #The way it is, observation just have to be a shape
        # ic(obs)
        obs=np.array(obs,np.float64)
        # ic(obs)
        # Add a batch dimension to the observation (e.g., from shape [obs_dim] to [1, obs_dim])
        obs = np.expand_dims(obs, axis=0)
        # ic(obs)
        if th.cuda.is_available():
            obs = th.tensor(obs).cuda()
        else:
            obs = th.tensor(obs)
        # ic(obs)
        # obs=model.policy.obs_to_tensor(obs)[0]
        # ic(obs)
        act = self.actor(obs, deterministic = True)
        # ic(act)
        action = act.cpu().detach().numpy().squeeze()
        # ic(action)
        # action += np.random.normal(0,0.1)
        return action
