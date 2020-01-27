from collections import deque
import numpy as np
import os
from IPython import embed

class Memory(object):
    '''
    database is deque object whose items are lists which
    contain ['current_state', 'action', 'next_state', 'reward']
    '''
    def __init__(self, size=None):
        if os.path.exists("../Database/episodes.npz"):
            loaded_object = np.load("../Database/episodes.npz", allow_pickle=True)
            self.__database = loaded_object['database']
        else:
            self.__database = deque(maxlen=size)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.__database), size=batch_size)
        samples = [self.__database[indices[i]] for i in range(batch_size)]
        return samples
    
    def append(self, sample):
        self.__database.append(sample)
    
    def save_database(self):
        np.savez("../Database/episodes.npz", database=self.__database)
    
    @property
    def database(self):
        return self.__database
    @property
    def length(self):
        return len(self.__database)