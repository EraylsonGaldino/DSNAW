import pandas as pd
import numpy as np

"""
separa val2 p selecionar o K

"""


import pickle
np.random.seed(42)
from preprocessamento import *  
from sklearn.svm import SVR
import itertools
from sklearn.metrics import mean_squared_error as MSE

def train_svr(x_train, y_train, x_val, y_val):
    
    melhor_mse = np.Inf 
    kernel = ['rbf']
    gamma = [0.5]#, 0.1, 10], 20, 30, 40, 50,60,70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    eps = [1]#, 0.1, 0.01, 0.001], 0.0001, 0.00001, 0.000001]
    C = [0.1]#, 1, 100, 1000, 10000]
    hyper_param = list(itertools.product(kernel, gamma, eps, C))
    
    for k, g, e, c in hyper_param:
        modelo = SVR(kernel=k,gamma=g, epsilon=e, C=c )
        modelo.fit(x_train, y_train)
        prev_v = modelo.predict(x_val)
        novo_mse  = MSE(y_val, prev_v)
        if novo_mse < melhor_mse:
            melhor_mse = novo_mse
            melhor_modelo = modelo

    return melhor_modelo, melhor_mse


def reamostragem(serie, n):
    size = len(serie)
    #nova_particao = []
    ind_particao = []
    for i in range(n):
        ind_r = np.random.randint(size)
        ind_particao.append(ind_r)
        #nova_particao.append(serie[ind_r,:])
    
    return ind_particao

def bagging(qtd_modelos, X_train, y_train, lags_acf):
    
    ens = []
    ensemble = {'models':[], 'indices': [] }   
    ind_particao = []
    
    if len(y_train.shape) == 1:
        y_train =  y_train.reshape(len(y_train), 1)
    
    
    train = np.hstack([X_train, y_train])
    
    for i in range(qtd_modelos):
        
        print('Training model: ', i)
        tam = len(train)
       
        indices = reamostragem(train, tam)
        
        particao = train[indices, :]
        
        
        Xtrain, Ytrain = particao[:, 0:-1], particao[:, -1]
        tam_val = int(len(Ytrain)*0.32)
        x_train = Xtrain[0:-tam_val, lags_acf]
        y_train = Ytrain[0:-tam_val]
        x_val = Xtrain[-tam_val:, lags_acf]
        y_val = Ytrain[-tam_val:]
        
        
        model, _ = train_svr(x_train, y_train, x_val, y_val)
        #return modelo
        ens.append(model)
        ind_particao.append(indices)
        
    
    
    ensemble['models'] = ens
    ensemble['indices'] = ind_particao
    
   
    return ensemble

def split_train_val_test(serie, p_tr, p_v1, p_v2):
    tam_serie =  len(serie)
    #print(tam_serie)
    tam_train = round(p_tr * tam_serie)
    tam_val1 = round(p_v1 * tam_serie)
    tam_val2 = round(p_v2 * tam_serie)
    #tam_test = tam_serie - (tam_train +  tam_val1 + tam_val2 )



    return  serie[0:tam_train] , serie[tam_train:tam_train+tam_val1] , serie[tam_train+tam_val1:tam_train+tam_val1+tam_val2] , serie[tam_train+tam_val1+tam_val2: ]


serie_name = 'star'
caminho = f'https://raw.githubusercontent.com/EraylsonGaldino/dataset_time_series/master/{serie_name}.txt'
print('Série:', serie_name)
dados = pd.read_csv(caminho, delimiter=' ', header=None)

serie = dados[0].values
serie_normalizada = normalise(serie)
p_tr = 0.50 #50% treinamento
p_val1 = 0.10 #10% val1: gridsearch
p_val2 = 0.15 #15% val2: select k 


train, val1, val2, test = split_train_val_test(serie_normalizada, p_tr, p_val1, p_val2)
#train, test = split_serie_less_lags(serie_normalizada, 0.75)
#no bagging é validação é selecionada após a remostragem. por isso, junta o train e val1 
train = np.hstack([train, val1])

max_lag = 20
lags_acf = select_lag_acf(serie_normalizada, max_lag)
max_sel_lag = lags_acf[0]
train_lags = create_windows(train, max_sel_lag+1)
test_lags = create_windows(test, max_sel_lag+1)
tam_val = len(val1)

X_train, y_train = train_lags[:, 0:-1], train_lags[:, -1]
ensemble = bagging(10, X_train, y_train, lags_acf)

ensemble_condig = {'ensemble': ensemble['models'], 'acf': lags_acf}
nome_arquivo = 'models\\'+serie_name+'_svr_pool.pkl'
pickle.dump( ensemble_condig, open( nome_arquivo, "wb" ), protocol=pickle.HIGHEST_PROTOCOL )

