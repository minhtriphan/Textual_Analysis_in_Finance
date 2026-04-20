class Config():
    def __init__(self):
        # For data handling
        self.max_length = 32
        # For training
        self.nepochs = 1000
        self.batch_size = 4
        self.num_workers = 1
        self.lr = 1e-3

    def __str__(self):
        return str(self.__dict__)
