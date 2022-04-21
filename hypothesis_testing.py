def hypothesis_testing_lm(dat, X, lambda_value):
    hp = {}
    d = X.shape[1]
    p = dat.Y.shape[1]
    effective_degree_freedom = dat.Y.shape[0] - d
    A = np.dot(dat.Y.T, X).T
    hp['B'] = compute_B_ridge(A, X, lambda_value)
    hp['epsilon_sigma2'] = dat.sigma2_lm(X, hp['B'], effective_degree_freedom)
    aux = np.linalg.solve(np.dot(X.T,X) +\
                np.diag(lambda_value*np.ones(d)), np.identity(d))
    hp['B_sigma2'] = np.dot(np.expand_dims(np.diag(aux), 1),
                      np.expand_dims(hp['epsilon_sigma2'],0)).T
    hp['score'] = hp['B'] / np.sqrt(hp['B_sigma2'])
    hp['pvalue'] = compute_pvalue_from_tscore(hp['score'],
                                              df=effective_degree_freedom)
    return hp