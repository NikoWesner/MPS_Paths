class MPO:

    def __init__(self,expo):
        self.cores=[None]*expo
        self.expo=expo
        self.p=1
        self.E=0
        self.r=[None]*(expo-1)

