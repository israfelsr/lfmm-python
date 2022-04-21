class RidgeLFMM:
    def __init__(self, K, lambda_value, algorithm="analytical"):
        self.K = K
        self.lambda_value = lambda_value
        self.algorithm = algorithm
    
    def lfmm_fit(self, dat, it_max=100, relative_err_min=1e-6):
        assert self.algorithm in ["analytical", "alternated"], f"algorithm must"\
            f"be analytical or alternated"
        if dat.missing_idx:
            assert self.algorithm != "analytical", f"Exact method doesn't allow"\
                f"missing data. Use an imputation method before running lfmm."
            self.fit_withNA(dat, it_max, relative_err_min)
        if self.algorithm == "analytical":
            self.fit_noNA(dat)
        if self.algorithm == "alternated":
            self.fit_noNa_alternated()

    def fit_withNA(self, dat, it_max=100, relative_err_min=1e-6): pass

    def fit_noNA(self, dat):
        pmatrix = compute_pmatrix(dat.X, self.lambda_value)
        self.ridge_lfmm(dat, pmatrix)

    def fit_noNa_alternated(self, dat, it_max=100, relative_err_min=1e-6): pass

    def ridge_lfmm(self, dat, pmatrix):
        _, d = dat.X.shape
        n, p = dat.Y.shape
        u, d, v = compute_svds(pmatrix[0], dat.Y, self.K)
        U = np.dot(u, np.diag(d))
        self.U = np.dot(pmatrix[1], U)
        self.V = v
        A = np.dot(dat.Y.T, dat.X).T - np.dot(np.dot(dat.X.T, self.U), self.V.T)
        self.B = compute_B_ridge(A, dat.X, self.lambda_value)