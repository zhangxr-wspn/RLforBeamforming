from __future__ import division
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class RayleighEnvWithoutCache(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):

        self.Nt = 2
        self.K_User = 2
        self.M_Group = 2
#        self.max_length=10**6
#        self.max_distance=3 # maximum distance 3km
        
        self.high_fading = np.array(10*np.ones(2*self.Nt*self.K_User*self.M_Group),dtype='float32') # fading
        
#        self.high_length = np.array([self.max_length, self.max_length]) # length
#        self.low_length = np.array([0,0]) #l1_packet,l2_packet
        
        self.high_action = np.array(np.append([np.ones(self.Nt*self.M_Group*2)],[1])) #w1(Nt*2),w2(Nt*2),rho        
        self.low_action = np.array(np.append([-1*np.ones(self.Nt*self.M_Group*2)],[0]))     

        self.observation_space = spaces.Box(low=-self.high_fading,high=self.high_fading,dtype='float32')
        self.action_space = spaces.Box(low=self.low_action, high=self.high_action, dtype='float32')

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):

#        l1_packet, l2_packet = self.state[-2:] # number of packet

        u_beamformer = u[0:self.Nt*self.M_Group*2]/((np.sum(u[0:self.Nt*self.M_Group*2]**2))**0.5) # beamformer unified power
        u_unify = np.append(u_beamformer,u[-1]) # added power ratio
    
#        self.last_u = u_unify # for rendering
        
        E_transmit = 10 # transmission power
        EtoN = E_transmit/self.Nt # transmission power
#        sigma = 0.01
        sigma = 10**(-5.4) # noise power
###########################Path loss########################  

        distance = np.array([0.02,0.05,0.02,0.05],dtype=float) # distance for user_m,k (11,12,21,22) , in km
#        beta = np.array([10,20,30,40],dtype=float) # path loss for user_m,k (11,12,21,22)
        beta = np.array(10**((-128.1-37.6*np.log10(distance))/10))
    
##########################Beamformer########################    
        w = np.array(np.zeros([self.Nt,self.M_Group]),dtype=complex) # Nt*M complex beamformer matrix
        for m in range(self.M_Group):
            for nt in range(self.Nt):
                w[nt,m] = np.array([u_beamformer[m*2*self.Nt+nt*2] + u_beamformer[m*2*self.Nt+nt*2+1]*1j])
        
        
        r1 = u_unify[-1] # power ratio
        r2 = 1-r1
        r=np.array([r1,r2])
########################Channel matrix#########################        
#        h=np.random.randn(self.Nt,self.K_User*self.M_Group)\
#            +np.multiply(1j,np.random.randn(self.Nt,self.K_User*self.M_Group)) # channel fading for 2 users in 2 groups
        h=np.zeros([self.Nt,self.K_User*self.M_Group], dtype='complex')# channel fading for 2 users in 2 groups
        index_temp=0
        for m in range(self.M_Group):
            for k in range(self.K_User):
                for nt in range(self.Nt):
                    h[nt,m*self.M_Group+k]=\
                    np.array(self.state[index_temp]+1j*self.state[index_temp+1])
                    index_temp+=2
        g=np.array(h)
                
        for nt in range(self.Nt):
            g[nt,:]=np.multiply(h[nt,:],beta) # fast fading multiply path loss
        
        #g=np.array([np.multiply(h[0,:],beta), np.multiply(h[1,:],beta)]) # 2*1 channel matrix, g11,g12,g21,g22
        
        sinr=np.array(np.zeros([2,2]),dtype='float32')
        
        sinr[0,0]=r[0]*EtoN*np.linalg.norm(np.matmul(g[:,0:1].conj().T,w[:,0:1]))\
            /(r[1]*EtoN*np.linalg.norm(np.matmul(g[:,0:1].conj().T,w[:,1:2]))+sigma**2)
        sinr[0,1]=r[0]*EtoN*np.linalg.norm(np.matmul(g[:,1:2].conj().T,w[:,0:1]))\
            /(r[1]*EtoN*np.linalg.norm(np.matmul(g[:,1:2].conj().T,w[:,1:2]))+sigma**2)
        sinr[1,0]=r[1]*EtoN*np.linalg.norm(np.matmul(g[:,2:3].conj().T,w[:,1:2]))\
            /(r[0]*EtoN*np.linalg.norm(np.matmul(g[:,2:3].conj().T,w[:,0:1]))+sigma**2)
        sinr[1,1]=r[1]*EtoN*np.linalg.norm(np.matmul(g[:,3:4].conj().T,w[:,1:2]))\
            /(r[0]*EtoN*np.linalg.norm(np.matmul(g[:,3:4].conj().T,w[:,0:1]))+sigma**2)
            
            
#        print("############\n")
#        print("state:", self.state,"u:",u,"w:",w,"r:",r,"g:",g,"sinr:",sinr)
        
#        B=10**7 #bandwith 
#        QAM=4 #QAM Order
        
#        mu=np.array([B*np.log2(1+sinr[0,:2].min()),B*np.log2(1+sinr[1,:2].min())])/np.log2(QAM) # Blog2(1+SINR) [bps] / (bit/packet)
#        lmbda = np.array([5*10**6,5*10**6],dtype=float) # Arrival rate
#        print(mu)
#        costs = (0.5*l1_packet/lmbda[0]+0.5*l2_packet/lmbda[1])*10**6 # delay [us]
#        if sinr.min() == 0:costs=100000
#        else: costs = sinr.min()**(-1)
        costs = 10**(-sinr.min())
#        rnd = np.random.rand()
#        newl1_packet = (l1_packet+1)*np.bool(rnd<=lmbda[0]/(lmbda[0]+mu[0]))+(l1_packet-1)*np.bool(rnd>lmbda[0]/(lmbda[0]+mu[0]))
#        newl2_packet = (l2_packet+1)*np.bool(rnd<=lmbda[1]/(lmbda[1]+mu[1]))+(l2_packet-1)*np.bool(rnd>lmbda[1]/(lmbda[1]+mu[1]))
        
#        if newl1_packet>=self.max_length:newl1_packet=self.max_length
#        elif newl1_packet<=0:newl1_packet=0
#        if newl2_packet>=self.max_length:newl2_packet=self.max_length
#        elif newl2_packet<=0:newl2_packet=0


#        self.state = np.array([newl1_packet, newl2_packet])
        self.state = self.np_random.randn(2*self.Nt*self.K_User*self.M_Group)
        
        
        return self.state, -costs, False, {}
###########################pass in the Nt number##################
    def reset(self):
        self.state = self.np_random.randn(2*self.Nt*self.K_User*self.M_Group)# Initial length
        self.last_u = None
        return self.state

    def close(self):
        if self.viewer: self.viewer.close()
