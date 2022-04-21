def lfmm_ridge(Y, X, K, lambda_value=1e-5, algorithm="analytical",
               it_max=100, relative_err_min=1e-6):
    m = RidgeLFMM(K, lambda_value, algorithm)
    dat = LfmmDat(center_columns(Y), center_columns(X))
    m.lfmm_fit(dat, it_max = it_max, relative_err_min = relative_err_min)
    return m

def lfmm_test(Y, X, lfmm, calibrate="gif"):
    dat = LfmmDat(center_columns(Y), center_columns(X))
    X = np.c_[dat.X, lfmm.U]
    d = dat.X.shape[1]
    #TODO: implement support to not-ridge lfmm
    lambda_value = lfmm.lambda_value
    hp = hypothesis_testing_lm(dat, X, lfmm.lambda_value)
    hp['score'] = hp['score'][:,0]
    hp['pvalue'] = hp['pvalue'][:,0]
    hp['B'] = hp['B'][:,0]
    # TODO: add support for other calibrations
    if calibrate == "gif":
        hp['gif'] = compute_gif(hp['score'])
        hp['calibrated_score2'] = (hp['score']**2) / hp['gif']
        hp['calibrated_pvalue'] = compute_pvalue_from_zscore2(hp['calibrated_score2'], df=1)
    return hp