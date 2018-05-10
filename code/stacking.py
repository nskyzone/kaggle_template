
# coding: utf-8

# # 训练模型

# In[2]:


import pandas as pd
import numpy as np
import xgboost
import lightgbm


# In[ ]:


from sklearn.model_selection import StratifiedKFold

def get_out_fold(model, x_train, y_train, x_predict,SEED=2018, NFOLDS=5, fit_params={}):
    kf = StratifiedKFold(n_splits = NFOLDS, random_state=SEED, shuffle=True)
    oof_train = np.zeros((x_train.shape[0],))
    oof_predict = np.zeros((x_predict.shape[0],))
    oof_predict_skf = np.empty((NFOLDS, x_predict.shape[0]))

    for i, (train_index, predict_index) in enumerate(kf.split(x_train,y_train)):
        x_tr = x_train.Discuss_seq[train_index]
        y_tr = y_train[train_index]
        x_te = x_train.Discuss_seq[predict_index]

        # clf.fit(x_tr, y_tr)
        # model.reset_states()
        model.fit(x_tr, y_tr, **fit_params)

        oof_train[predict_index] = model.predict(x_te).squeeze()
        oof_predict_skf[i, :] = model.predict(x_predict).squeeze()

    # oof_predict[:] = oof_predict_skf.mean(axis=0)
    return oof_train, oof_predict_skf.mean(axis=0)

# 先用100样本快速测试
# fit_params={}
# cnn_oof_train, cnn_oof_test = get_out_fold(model,x_train,y_train,x_predict)


# In[ ]:


from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier,BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
def gen_model():
    rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt',max_depth=6, 
                                min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)
    ridge = Ridge(solver='auto', fit_intercept=True, alpha=1, max_iter=250, normalize=False, tol=0.01)

    ada = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)

    et = ExtraTreesRegressor(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)

    gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)

    dt = DecisionTreeRegressor(max_depth=8)

    knn = KNeighborsRegressor(n_neighbors = 2)

    svm = SVR(kernel='linear', C=0.025)
    return rf,ridge,ada,et,gb,dt,knn


# In[3]:



def train_model(rf,ridge,ada,et,gb,dt,knn):
    ridge_oof_train, ridge_oof_test = get_out_fold(ridge, x_train, y_train, x_predit) # Ridge
    rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_predit) # Random Forest
    ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_predit) # AdaBoost 
    et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_predit) # Extra Trees
    gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_predit) # Gradient Boost
    dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_predit) # Decision Tree
    knn_oof_train, knn_oof_test = get_out_fold(knn, x_train, y_train, x_predit) # KNeighbors
    svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_predit) # Support Vector
    return (ridge_oof_train,rf_oof_train,ada_oof_train,gb_oof_train,dt_oof_train),(ridge_oof_test,rf_oof_test,ada_oof_test,gb_oof_test,dt_oof_test)


# # 加载cv数据作为训练

# In[ ]:


ridge_oof_cv = pd.read_csv('../data/ridge_oof_cv.csv')
ridge_oof_predict = pd.read_csv('../data/ridge_oof_predict.csv')

ada_oof_cv = pd.read_csv('../data/ada_oof_cv.csv')
ada_oof_predict = pd.read_csv('../data/ada_oof_predict.csv')

rf_oof_cv = pd.read_csv('../data/rf_oof_cv.csv')
rf_oof_predict = pd.read_csv('../data/rf_oof_predict.csv')

gb_oof_cv = pd.read_csv('../data/gb_oof_cv.csv')
gb_oof_predict = pd.read_csv('../data/gb_oof_predict.csv')

dt_oof_cv = pd.read_csv('../data/dt_oof_cv.csv')
dt_oof_predict = pd.read_csv('../data/dt_oof_predict.csv')

svm_oof_cv = pd.read_csv('../data/svm_oof_cv.csv')
svm_oof_predict = pd.read_csv('../data/svm_oof_predict.csv')

cnn_oof_cv = pd.read_csv('../data/cnn_oof_cv.csv')
cnn_oof_predict = pd.read_csv('../data/cnn_oof_predict.csv')

rnn_oof_cv = pd.read_csv('../data/rnn_oof_cv.csv')
rnn_oof_predict = pd.read_csv('../data/rnn_oof_predict.csv')


# In[ ]:


x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, knn_oof_train, svm_oof_train), axis=1)
x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, knn_oof_test, svm_oof_test), axis=1)


# In[ ]:


gbm = xgboost.XGBRegressor( n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8, 
                        colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)

