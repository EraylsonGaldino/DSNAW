import pandas as pd
import numpy as np



import pickle
np.random.seed(42)
from preprocessamento import *  
from sklearn.svm import SVR
import itertools
from sklearn.metrics import mean_squared_error as MSE

def train_svr(x_train, y_train, x_val, y_val):
    
    melhor_mse = np.Inf 
    kernel = ['rbf']
    gamma = [0.5, 0.1, 10]#, 20, 30, 40, 50,60,70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    eps = [1, 0.1, 0.01, 0.001]#, 0.0001, 0.00001, 0.000001]
    C = [0.1, 1, 100, 1000, 10000]#
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

def split_train_val_test(series, p_tr, perc_val = 0):
    tam_serie =  len(series)
    #print(tam_serie)
    train_size = int(np.ceil(p_tr * tam_serie))
    
    if perc_val > 0:
        
        val_size = int(np.ceil(len(serie) *perc_val))
        
        
        
        x_train = series[0:train_size]
        x_val = series[train_size:train_size+val_size]        
        x_test = series[(train_size+val_size):]
        
        return x_train, x_test, x_val
        
    else:
        
                
        x_train = series[0:train_size]
        x_test = series[train_size:]
        

        return x_train, x_test

def desempenho_media_pool(ensemble, x_test, y_test):
    previsao = []
    for janela in x_test:
        previsoes = []
        for modelo in ensemble:
            prev = modelo.predict(janela.reshape(1, -1))
            previsoes.append(prev)
        previsao.append(np.mean(previsoes))

    print(MSE(y_test, previsao))
    


serie_name = 'APPLE'
caminho = f'https://raw.githubusercontent.com/EraylsonGaldino/dataset_time_series/master/{serie_name}.txt'
print('Série:', serie_name)
dados = pd.read_csv(caminho, delimiter=' ', header=None)

serie = dados[0].values
serie_normalizada = normalise(serie)
p_tr = 0.75 #75% treinamento



train, test = split_train_val_test(serie_normalizada, p_tr)
#train, test = split_serie_less_lags(serie_normalizada, 0.75)
#no bagging é validação é selecionada após a remostragem. por isso, junta o train e val1 


max_lag = 20
lags_acf = select_lag_acf(serie_normalizada, max_lag)
max_sel_lag = lags_acf[0]
train_lags = create_windows(train, max_sel_lag+1)
test_lags = create_windows(test, max_sel_lag+1)

X_train, y_train = train_lags[:, 0:-1], train_lags[:, -1]
ensemble = bagging(100, X_train, y_train, lags_acf)
x_test, y_test = test_lags[:, 0:-1], test_lags[:, -1]
desempenho_media_pool(ensemble['models'], x_test[:, lags_acf], y_test)
ensemble_condig = {'ensemble': ensemble['models'], 'acf': lags_acf}
nome_arquivo = 'models\\'+serie_name+'_svr_pool.pkl'
pickle.dump( ensemble_condig, open( nome_arquivo, "wb" ), protocol=pickle.HIGHEST_PROTOCOL )

