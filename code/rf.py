def rf(X_train, X_test, y_train, y_test , max_depth):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    import pandas as pd
    import numpy as np
    r2s = []

    for i in range(len(max_depth)):
        rf = RandomForestRegressor(max_depth = max_depth[i], n_jobs = -1,oob_score = True, bootstrap = True, random_state = 42)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        r2s.append(r2_score(y_test, pred))
    best_depth = max_depth[np.argmax(r2_score)]
    rf = RandomForestRegressor(max_depth = best_depth, n_jobs = -1,oob_score = True, bootstrap = True, random_state = 42).fit(X_train, y_train)
    impt = rf.feature_importances_
    perm = pd.DataFrame({'features': X_train.columns, 'importance': impt}).sort_values('importance', ascending=False)
    best_features = perm[perm['importance']>0.05]['features'].to_numpy()
    return best_features