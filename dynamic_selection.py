import pandas as pd
import numpy as np
#from sklearn.externals import joblib
import  warnings 
warnings.simplefilter('ignore')
from sklearn.metrics import mean_squared_error as MSE
from preprocessamento import *  
import pickle

from select_approaches import dsnaw

serie_name = 'star'
caminho = f'https://raw.githubusercontent.com/EraylsonGaldino/dataset_time_series/master/{serie_name}.txt'
print('Série:', serie_name)
dados = pd.read_csv(caminho, delimiter=' ', header=None)

serie = dados[0].values
serie_normalizada = normalise(serie)
train, test = split_serie_less_lags(serie_normalizada, 0.8)
max_lag = 20
lags_acf = select_lag_acf(serie_normalizada, max_lag)
max_sel_lag = lags_acf[0]
#(input, target)
train_lags = create_windows(train, max_sel_lag+1)

test = np.hstack([train[-max_lag:], test])
test_lags = create_windows(test, max_sel_lag+1)

X_test = test_lags[:, lags_acf ]
y_test = test_lags[:,  -1]

previous_data = train_lags

nome_arquivo = 'models\\'+serie_name+'_svr_pool.pkl'
ensemble = pickle.load(open( nome_arquivo, "rb" ))
ensemble = ensemble['ensemble']
m = 1
k = 10
previsoes_dsnaw = []
for i in range(len(y_test)):
    #selection by dsnaw
    ind_models = dsnaw(previous_data, ensemble,lags_acf, k, m )
    
    if len(ind_models) == 1:
        model_selected = ensemble[ind_models[0]]
        prev = model_selected.predict(X_test[i].reshape(1, -1))[0]
        
    else:
        ensemble_selected = ensemble[ind_models]
         #utilizando a média
        previsoes_ensemble = []
        for modelo in ensemble_selected:
            prev = modelo.predict(X_test[i].reshape(1, -1))
            previsoes_ensemble.append(prev)
        prev = np.mean(previsoes_ensemble)
    
    previsoes_dsnaw.append(prev)


breakpoint()


    




