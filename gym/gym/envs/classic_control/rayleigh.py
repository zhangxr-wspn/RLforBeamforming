from __future__ import division
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class RayleighEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):


        self.Nt = 32 # Directly convergence
#        self.Nt = 2
        self.K_User = 2
        self.M_Group = 2
#        self.max_length=10**6
        self.max_length=10**6
        
        
        high_length = np.array([self.max_length, self.max_length]) #l1_packet,l2_packet
        low_length = np.array([0,0]) #l1_packet,l2_packet
        
        high_action = np.array(np.append([np.ones(self.Nt*self.M_Group*2)],[1])) #w1(Nt*2),w2(Nt*2),rho        
        low_action = np.array(np.append([-1*np.ones(self.Nt*self.M_Group*2)],[0]))     

        self.observation_space = spaces.Box(low=low_length, high=high_length)
        self.action_space = spaces.Box(low=low_action, high=high_action)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):

        l1_packet, l2_packet = self.state # number of packet
        #print(u)
        #u_unify = u[0]/((np.sum(u[0]**2))**0.5)
#        u_unify = u/((np.sum(u**2))**0.5)
        u_beamformer = u[0:self.Nt*self.M_Group*2]/((np.sum(u[0:self.Nt*self.M_Group*2]**2))**0.5) # beamformer unified power
        u_unify = np.append(u_beamformer,u[-1]) # added power ratio
    
        self.last_u = u_unify # for rendering
        
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
                #w[nt,m:m+1] = np.array([[u_unify[m*4]+u_unify[m*4+1]*1j],[u_unify[m*4+2]+u_unify[m*4+3]*1j]])
        #        print(m*2*self.Nt+nt*2)
        #        print(u_unify[0])
                w[nt,m] = np.array([u_beamformer[m*2*self.Nt+nt*2] + u_beamformer[m*2*self.Nt+nt*2+1]*1j])
        # u = [u0,u1,u2,u3,u4.....]
        # w = [ u0+ju1   # nt=0
            # u2+ju3   # nt=1
            # ...     #...
            # group m=0
        #w[:,0:1] = np.array([[u_unify[0]+u_unify[1]*1j],[u_unify[2]+u_unify[3]*1j]]) # 2*1 beamformer
        #w[:,1:2] = np.array([[u_unify[4]+u_unify[5]*1j],[u_unify[6]+u_unify[7]*1j]])
        
        r1 = u_unify[-1] # power ratio
        r2 = 1-r1
        r=np.array([r1,r2])
########################Channel matrix#########################        
        h=np.random.randn(self.Nt,self.K_User*self.M_Group)\
            +np.multiply(1j,np.random.randn(self.Nt,self.K_User*self.M_Group)) # channel fading for 2 users in 2 groups
        g=np.array(h)
                
        for nt in range(self.Nt):
            g[nt,:]=np.multiply(h[nt,:],beta) # fast fading multiply path loss
        
        #g=np.array([np.multiply(h[0,:],beta), np.multiply(h[1,:],beta)]) # 2*1 channel matrix, g11,g12,g21,g22
        
        sinr=np.array(np.zeros([2,2]))
                
#        sinr[0,0]=r[0]*EtoN*np.absolute(np.matmul(g[:,0:1].conj().T,w[:,0:1]))\
#            /(r[1]*EtoN*np.absolute(np.matmul(g[:,0:1].conj().T,w[:,1:2]))+sigma**2)
#        sinr[0,1]=r[0]*EtoN*np.absolute(np.matmul(g[:,1:2].conj().T,w[:,0:1]))\
#            /(r[1]*EtoN*np.absolute(np.matmul(g[:,1:2].conj().T,w[:,1:2]))+sigma**2)
#        sinr[1,0]=r[1]*EtoN*np.absolute(np.matmul(g[:,2:3].conj().T,w[:,1:2]))\
#            /(r[0]*EtoN*np.absolute(np.matmul(g[:,2:3].conj().T,w[:,0:1]))+sigma**2)
#        sinr[1,1]=r[1]*EtoN*np.absolute(np.matmul(g[:,3:4].conj().T,w[:,1:2]))\
#            /(r[0]*EtoN*np.absolute(np.matmul(g[:,3:4].conj().T,w[:,0:1]))+sigma**2)
            
        #?????    too many 0????
#        print("sinr[0,0] numerator",r[0],EtoN,np.linalg.norm(np.matmul(g[:,0:1].conj().T,w[:,0:1])))
        
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
        
        B=10**7 #bandwith 
        QAM=4 #QAM Order
        
        mu=np.array([B*np.log2(1+sinr[0,:2].min()),B*np.log2(1+sinr[1,:2].min())])/np.log2(QAM) # Blog2(1+SINR) [bps] / (bit/packet)
        lmbda = np.array([5*10**6,5*10**6],dtype=float) # Arrival rate
#        print(mu)
        costs = (0.5*l1_packet/lmbda[0]+0.5*l2_packet/lmbda[1])*10**6 # delay [us]
#        costs = -sinr.min()
        rnd = np.random.rand()
        newl1_packet = (l1_packet+1)*np.bool(rnd<=lmbda[0]/(lmbda[0]+mu[0]))+(l1_packet-1)*np.bool(rnd>lmbda[0]/(lmbda[0]+mu[0]))
        newl2_packet = (l2_packet+1)*np.bool(rnd<=lmbda[1]/(lmbda[1]+mu[1]))+(l2_packet-1)*np.bool(rnd>lmbda[1]/(lmbda[1]+mu[1]))
        
        if newl1_packet>=self.max_length:newl1_packet=self.max_length
        elif newl1_packet<=0:newl1_packet=0
        if newl2_packet>=self.max_length:newl2_packet=self.max_length
        elif newl2_packet<=0:newl2_packet=0


        self.state = np.array([newl1_packet, newl2_packet])

        return self._get_obs(), -costs, False, {}

    def reset(self):
 #       high = np.array([np.pi, 1])
 #       self.state = self.np_random.uniform(low=-high, high=high)
#        self.state = np.array([self.max_length/2,self.max_length/2]) # Initial length
        self.state = np.array([30,30]) # Initial length
        self.last_u = None
 #       return np.array([10, 20])
        return self._get_obs()

    def _get_obs(self):
        l1_packet, l2_packet = self.state
        return np.array([l1_packet, l2_packet])
   #     return np.array([10, 20])

############################rendering#####################################
    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

#def angle_normalize(x):
#    return (((x+np.pi) % (2*np.pi)) - np.pi)
