class LfmmDat:
    def __init__(self, Y, X, missing=True):
        self.Y = Y
        self.X = X
        self.missing_idx = list(map(tuple,np.argwhere(np.isnan(x))))
    
    def sigma2_lm(self, X, B, nb_df):
        return sum2_lm(self.Y, X, B) / nb_df