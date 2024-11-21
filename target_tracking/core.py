import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import math
from scipy.integrate import odeint
from icecream import ic


class PlayerParticle():
    # state = [x,y]
    def __init__(self, player_speed, env_size, dt, player_size, winning_radius):
        self.dt = dt
        self.player_size = player_size # Rendering info only
        self.winning_radius = winning_radius
        
        self.min_action_angle = -np.pi
        self.max_action_angle = np.pi
        
        self.min_action_speed = 0
        self.max_action_speed = player_speed

        self.min_position = -env_size
        self.max_position = env_size

        self.low_state = np.array([self.min_position,self.min_position], dtype=np.float32)
        self.high_state = np.array([self.max_position,self.max_position], dtype=np.float32)

        self.min_action = np.array([self.min_action_angle], dtype=np.float32)
        self.max_action = np.array([self.max_action_angle], dtype=np.float32)

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
             shape=(1,), dtype=np.float32
        )
        
        self.state_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        #self.action_space = spaces.Box(
        #    low=self.min_action, high=self.max_action, shape=(2,), dtype=np.float32
        #)
        self.seed()
        
       
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepCoarse(self, action):
        #action = angle
        action = np.arctan2(np.sin(action), np.cos(action)) # This wraps action to -pi,pi
        
        old_pos = self.state[[0,1]]
        new_pos = old_pos + self.max_action_speed*np.array([math.cos(action),math.sin(action)])*self.dt
        new_pos[0] = np.clip(new_pos[0], self.min_position, self.max_position)
        new_pos[1] = np.clip(new_pos[1], self.min_position, self.max_position)
        
        self.state = np.array([new_pos[0], new_pos[1]], dtype=np.float32)
        return self.state

    def set_speed(self, speed):
        self.max_action_speed = speed
        return
    
    def get_speed(self):
        return self.max_action_speed 
    
    @staticmethod
    def derivative(y,t, action, speed):
        return speed*np.array([math.cos(action),math.sin(action)])
    
    def stepFine(self, action):
        y0 = self.state[[0,1]]
        t = np.array([0, self.dt])
        y = odeint(self.derivative, y0, t, args=(action,self.max_action_speed))
        self.state = y[-1]
        self.state = np.clip(self.state, self.min_position, self.max_position)
        return self.state


    def step(self,action, integration_mode = "coarse"):
        if(integration_mode == "coarse"):
            return self.stepCoarse(action)
        elif(integration_mode == "fine"):
            return self.stepFine(action)
        else:
            raise Exception("Integration mode undefined!")

    def reset(self, seed=None):
        self.seed(seed=seed)

        # state = [posx, posy, angle, speed]
        self.state = np.array([self.np_random.uniform(low=self.min_position, high=self.max_position),
        self.np_random.uniform(low=self.min_position, high=self.max_position)])
        return np.array(self.state, dtype=np.float32)

    def denorm_action(self, action):
        action_nom = np.copy(action)
        action_nom =  (action_nom+ 1)*self.max_action_angle + self.min_action_angle
        return action_nom

class EvaderP(PlayerParticle):
    def __init__(self, player_speed, env_size, dt, player_size, winning_radius):
        super().__init__(player_speed, env_size, dt, player_size, winning_radius)

     # evader always spawns in the outer region
    def reset(self):
        min_radius = 2*self.max_position/3
        max_radius = self.max_position

        angle = self.np_random.uniform(low=-np.pi, high=np.pi)
        radius = self.np_random.uniform(low=min_radius, high=max_radius)

        self.state = np.array([radius*np.cos(angle), radius*np.sin(angle)])
        return np.array(self.state, dtype=np.float32)

class PursuerP(PlayerParticle):
    def __init__(self, player_speed, env_size, dt, player_size, winning_radius):
        super().__init__(player_speed, env_size, dt, player_size, winning_radius)

    # Pursuer always spawns in the inner region
    def reset(self):
        min_radius = 0
        max_radius = 2*self.max_position/3

        angle = self.np_random.uniform(low=-np.pi, high=np.pi)
        radius = self.np_random.uniform(low=min_radius, high=max_radius)

        self.state = np.array([radius*np.cos(angle), radius*np.sin(angle)])
        return np.array(self.state, dtype=np.float32) 
    
class PlayerDubbins():
    # state = [x,y,theta]
    def __init__(self, player_speed, env_size, dt, player_size, winning_radius):
        self.dt = dt
        self.player_size = player_size # Rendering info only
        self.winning_radius = winning_radius
        
        self.min_action_omega = -1
        self.max_action_omega = 1
        
        self.min_angle = -np.pi
        self.max_angle = np.pi

        self.min_action_speed = 0
        self.max_action_speed = player_speed

        self.min_position = -env_size
        self.max_position = env_size

        self.low_state = np.array([self.min_position,self.min_position,self.min_angle], dtype=np.float32)
        self.high_state = np.array([self.max_position,self.max_position,self.max_angle], dtype=np.float32)

        self.min_action = np.array([self.min_action_omega], dtype=np.float32)
        self.max_action = np.array([self.max_action_omega], dtype=np.float32)

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
             shape=(1,), dtype=np.float32
        )
        
        self.state_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        #self.action_space = spaces.Box(
        #    low=self.min_action, high=self.max_action, shape=(2,), dtype=np.float32
        #)
        self.seed()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepCoarse(self, action):
        #action = angle
        action = np.arctan2(np.sin(action), np.cos(action)) # This wraps action to -pi,pi
        
        old_state = np.array(self.state[[0,1,2]])
        
        new_state = old_state + np.array(
            [self.max_action_speed*np.cos(old_state[2]), 
             self.max_action_speed*np.sin(old_state[2]), 
             action])*self.dt
        
        # if  new_state[0] < self.min_position or\
        #     new_state[0] > self.max_position or\
        #     new_state[1] < self.min_position or\
        #     new_state[1] > self.max_position :
        #     # outside the are, dont update
        #     new_state[0] = old_state[0]
        #     new_state[1] = old_state[1]
        #     # new_state[2] = old_state[2] + np.random.uniform(low=0, high=0.2)
        #     # ic(old_state[2],action)

        
        # new_state[0] = np.clip(new_state[0], self.min_position, self.max_position)
        # new_state[1] = np.clip(new_state[1], self.min_position, self.max_position)
        new_state[2] = np.arctan2(np.sin(new_state[2]), np.cos(new_state[2])) 
        # ic(old_state,new_state)

        self.state = np.array([new_state[0], new_state[1], new_state[2]], dtype=np.float32)
        return self.state
    
    def set_speed(self, speed):
        self.max_action_speed = speed
        return
    
    def get_speed(self):
        return self.max_action_speed 
    
    @staticmethod
    def derivative(y,t, action, speed):
        # return speed*np.array([math.cos(action),math.sin(action)])
        return np.array(
            [speed*math.cos(y[2]), 
             speed*math.sin(y[2]), 
             action])
    
    def stepFine(self, action):
        y0 = self.state[[0,1,2]]
        t = np.array([0, self.dt])
        y = odeint(self.derivative, y0, t, args=(action,self.max_action_speed))
        new_state = y[-1]
        # new_state = np.clip(new_state, self.min_position, self.max_position)

        new_state[0] = np.clip(new_state[0], self.min_position, self.max_position)
        new_state[1] = np.clip(new_state[1], self.min_position, self.max_position)
        new_state[2] = np.arctan2(np.sin(new_state[2]), np.cos(new_state[2]))
        
        self.state = np.array([new_state[0], new_state[1], new_state[2]], dtype=np.float32)
        return self.state

    def step(self,action, integration_mode = "coarse"):
        # ic(action)
        action*=np.pi #action comes normalized from -1 to 1
        if(integration_mode == "coarse"):
            return self.stepCoarse(action)
        elif(integration_mode == "fine"):
            return self.stepFine(action)
        else:
            raise Exception("Integration mode undefined!")

    def reset(self, seed=None):
        self.seed(seed=seed)

        # state = [posx, posy, angle, speed]
        self.state = np.array([
            self.np_random.uniform(low=self.min_position, high=self.max_position),
            self.np_random.uniform(low=self.min_position, high=self.max_position),
            self.np_random.uniform(low=-np.pi, high=np.pi)])
        return np.array(self.state, dtype=np.float32)

    def denorm_action(self, action):
        action_nom = np.copy(action)
        action_nom =  (action_nom+ 1)*self.max_action_omega + self.min_action_omega
        return action_nom

class EvaderD(PlayerDubbins):
    def __init__(self, player_speed, env_size, dt, player_size, winning_radius):
        super().__init__(player_speed, env_size, dt, player_size, winning_radius)

    # evader always spawns in the outer region
    def reset(self, seed=None):
        if (seed is not None): 
            super.seed(seed=seed)
        min_radius = self.max_position/2
        max_radius = self.max_position

        angle = self.np_random.uniform(low=0, high=np.pi/2)
        # angle = self.np_random.uniform(low=-(3/8)*np.pi, high=(3/8)*np.pi)
        radius = self.np_random.uniform(low=min_radius, high=max_radius)
        
        self.state = np.array([
            radius*np.cos(angle), 
            radius*np.sin(angle), 
            0
            ]) 
        return np.array(self.state, dtype=np.float32)

class PursuerD(PlayerDubbins):
    def __init__(self, player_speed, env_size, dt, player_size, winning_radius):
        super().__init__(player_speed, env_size, dt, player_size, winning_radius)

    # Pursuer always spawns in the inner region
    def reset(self, seed=None):
        if (seed is not None): 
            super.seed(seed=seed)
        
        min_radius = 0
        max_radius = self.max_position

        angle = self.np_random.uniform(low=np.pi, high=3*np.pi/2)
        radius = self.np_random.uniform(low=min_radius, high=max_radius)

        self.state = np.array([
            radius*np.cos(angle), 
            radius*np.sin(angle), 
            self.np_random.uniform(low=-np.pi, high=np.pi)
            ])
        return np.array(self.state, dtype=np.float32)
    
class ExclusionZone():
    # state = [x,y]
    def __init__(self, radius, env_size, name):
        self.radius = radius
        self.min_position = -env_size/2 #+ radius
        self.max_position =  env_size/2 #- radius
        self.name = name 
        self.seed()
        self.reset()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None):
        if (seed is not None):
            self.seed(seed=seed)
        
        # pos = [posx, posy]
        self.pos = np.array([
            self.np_random.uniform(low=self.min_position, high=self.max_position),
            self.np_random.uniform(low=self.min_position, high=self.max_position)])

        return np.array(self.pos, dtype=np.float32)
    
    @staticmethod
    def EZs_overlap(EZ1,EZ2):
        if np.linalg.norm(EZ1.pos-EZ2.pos) < (EZ1.radius + EZ2.radius):
            return True
        return False
    
    @staticmethod
    def EZ_Agent_overlap(EZ1,agent):
        if np.linalg.norm(EZ1.pos-agent.state[0:2]) < (EZ1.radius):
            return True
        return False