from sklearn.model_selection import train_test_split
class TrainTestPerformer():

    def __init__(self, portion_train=0.8, random_sampling=True, random_sampling_seed=1):
        self.portion_train = portion_train
        self.random_sampling = random_sampling
        self.random_sampling_seed = random_sampling_seed
    
    def get(self, x, y):
        if(self.random_sampling is True):
            return train_test_split(x, y, train_size=self.portion_train, random_state=self.random_sampling_seed)
        
        temp_size = int(x.shape[0]*self.portion_train)
        return x[0:temp_size], x[temp_size::], y[0:temp_size], y[temp_size::]

