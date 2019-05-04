import numpy as np
import numpy.random as nr

class OUNoise:
    """docstring for OUNoise"""
    def __init__(self,action_dimension,mu=0., theta=0.023, sigma=0.02):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
        return self.state

    def noise(self):#, epi,order=0.):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = (x + dx)#/ np.power(epi+1,order)
        return self.state

if __name__ == '__main__':
    ou = OUNoise(1,theta=0.023,sigma=0.02)
    states = []
    print(ou.reset())
    for i in range(20000):
        states.append(ou.noise())
        # print(ou.noise())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
