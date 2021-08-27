#!/usr/bin/env python
# coding: utf-8

# In[14]:


from collections import deque
from collections import namedtuple
import numpy as np
import torch.optim as optimizer
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import random

# In[15]:


class Agent():
    
    def __init__(self,state_size,action_size,batch_size,buffer_size):
        self.local_critic=critic(state_size,action_size)
        self.target_critic=critic(state_size,action_size)
        self.local_policy=policy(state_size,action_size)
        self.target_policy=policy(state_size,action_size)
        self.memory=replay_buffer(batch_size,buffer_size)
        self.critic_optimizer=optimizer.Adam(self.local_critic.parameters(),lr=1e-3)
        self.policy_optimizer=optimizer.Adam(self.local_policy.parameters(),lr=1e-4)
        self.batch_size=batch_size
        self.noise=noise(4)
    def step(self,state,new_state,action,reward,done):
        self.memory.add(state,new_state,action,reward,done)
        if len(self.memory)>self.batch_size:
            experience=self.memory.sample()
            self.learn(experience)
    def learn(self,experience,gamma=.99):
        states,new_states,actions,rewards,dones=experience
        new_actions=self.target_policy(new_states)
        predicted_values=rewards+gamma*self.local_critic(new_states,new_actions)*(1-dones)
        expected_values=self.target_critic(states,actions)
        critic_loss=F.mse_loss(expected_values,predicted_values)
        self.critic_optimizer.zerod_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        new_actions_2=self.local_policy(states)
        policy_loss=-self.local_critic(states,new_actions_2).mean()
        self.policy_optimizer.zerod_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        self.update(self.local_policy,self.target_policy)
        self.update(self.local_critic,self.target_critic)
    def action(self,state,add_noise=True):
        state=torch.from_numpy(state).float()
        
        with torch.no_grad():
            action=self.local_policy(state).data.numpy()
        if add_noise:
            action=action+self.noise.sample()
        
        
        return np.clip(action,-1,1)
    def update(self,local_model,target_model,tau=.01):
        for local_param,target_param in zip(local_model.parameters(),target_model.parameters()):
            target_param.data.copy_(tau*local_param.data+(1-tau)*target_param.data)
                
            
        
        
        
        
        


# In[6]:


class critic(nn.Module):
    def __init__(self,state_size,action_size):
        super(critic,self).__init__()
        self.layer1=nn.Linear(state_size,128)
        self.layer2=nn.Linear(128+action_size,64)
        self.layer3=nn.Linear(64,32)
        self.layer4=nn.Linear(32,1)
    def forward(self,state,action):
        x=F.relu(self.layer1(state))
        x=torch.cat([x,action],dim=1)
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        return x


# In[13]:


class policy(nn.Module):
        def __init__(self,state_size,action_size):
            super(policy,self).__init__()
            self.layer1=nn.Linear(state_size,128)
            self.layer2=nn.Linear(128,64)
            self.layer3=nn.Linear(64,32)
            self.layer4=nn.Linear(32,action_size)
        def forward(self,state):
            x=F.relu(self.layer1(state))
            x=F.relu(self.layer2(x))
            x=F.relu(self.layer3(x))
            x=F.tanh(self.layer4(x))
            return x
    
    


# In[16]:


class noise():
    def __init__(self,action_size,theta=.15,mu=0,sigma=.2):
        self.theta=theta
        self.size=action_size
        self.mu=mu*np.ones(action_size)
        self.sigma=sigma
        self.reset()
    def reset(self):
        self.state=copy.copy(self.mu)
    def sample(self):
        x=self.state
        dx=self.theta*(x-self.mu)+self.sigma*np.array([random.random() for i in range(self.size)])
        self.state=x+dx
        return self.state
    
        


# In[17]:


class replay_buffer():
    def __init__(self,batch_size,buffer_size):
        self.memory=deque(maxlen=buffer_size)
        self.batch_size=batch_size
        self.experience=namedtuple("experience",field_names=["state","new_state","action","done","reward"])
    def add(self,state,new_state,action,reward,done):
        experience=self.experience(state,new_state,action,done,reward)
        self.memory.append(experience)
    def sample(self):
        experiences=random.sample(self.memory,self.batch_size)
        states=torch.from_numpy(np.vstack(e.state for e in experiences)).float()
        new_states=torch.from_numpy(np.vstack(e.new_state for e in experiences)).float()
        reward=torch.from_numpy(np.vstack(e.reward for e in experiences)).float()
        actions=torch.from_numpy(np.vstack(e.action for e in experiences)).float()
        done=torch.from_numpy(np.vstack(e.done for e in experiences)).float()
        return (states,new_states,actions,rewards,dones)
    def __len__(self):
        return len(self.memory)
        
    

