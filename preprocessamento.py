
import numpy as np


def normalise_interval(minimo, maximo, serie):
	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler(feature_range=(minimo, maximo))
	scaler = scaler.fit(serie)
	normalized = scaler.transform(serie)
	return normalized, scaler 
	
	
	
def desnorm_interval(serie_norm, serie_real, minimo, maximo):
	norm, scaler = normalise_interval(minimo, maximo, serie_real)
	inversed = scaler.inverse_transform(serie_norm)
	return inversed

def split_serie_with_lags(serie, perc_train, perc_val = 0):
    
    #faz corte na serie com as janelas já formadas 
    
    x_date = serie[:, 0:-1]
    y_date = serie[:, -1]        
       
    train_size = np.fix(len(serie) *perc_train)
    train_size = train_size.astype(int)
    
    if perc_val > 0:        
        val_size = np.fix(len(serie) *perc_val).astype(int)
              
        
        x_train = x_date[0:train_size,:]
        y_train = y_date[0:train_size]
        print("Particao de Treinamento:", 0, train_size  )
        
        x_val = x_date[train_size:train_size+val_size,:]
        y_val = y_date[train_size:train_size+val_size]
        
        print("Particao de Validacao:",train_size, train_size+val_size)
        
        x_test = x_date[(train_size+val_size):,:]
        y_test = y_date[(train_size+val_size):]
        
        print("Particao de Teste:", train_size+val_size, len(y_date))
        
        return x_train, y_train, x_test, y_test, x_val, y_val
        
    else:
        
        x_train = x_date[0:train_size,:]
        y_train = y_date[0:train_size]

        x_test = x_date[train_size:-1,:]
        y_test = y_date[train_size:-1]

        return x_train, y_train, x_test, y_test

def normalise(serie):
    minimo = min(serie)
    maximo = max(serie)
    y = (serie - minimo) / (maximo - minimo)
    return y


def desnorm(serie_atual, serie_real):
    import pandas as pd
    minimo = min(serie_real)
    maximo = max(serie_real)
    
    serie = (serie_atual * (maximo - minimo)) + minimo
    
    return list(serie) 


def create_windows(serie,tam_janela):
   # serie: vetor do tipo numpy ou lista
    tam_serie = len(serie)
    tam_janela = tam_janela +1 # Adicionado mais um ponto para retornar o target na janela
    
    janela = list(serie[0:0+tam_janela]) #primeira janela p criar o objeto np
    janelas_np = np.array(np.transpose(janela))    
       
    for i in range(1, tam_serie-tam_janela):
        janela = list(serie[i:i+tam_janela])
        j_np = np.array(np.transpose(janela))        
        
        janelas_np = np.vstack((janelas_np, j_np))
    
    
    janela = list(serie[i+1:tam_serie])
    
    j_np = np.array(np.transpose(janela)) 
    janelas_np = np.vstack((janelas_np, j_np))
    return janelas_np


def select_lag_acf(serie, max_lag):
    from statsmodels.tsa.stattools import acf
    x = serie[0: max_lag+1]
    
    acf_x, confint = acf(serie, nlags=max_lag, alpha=.05, fft=False,
                             unbiased=False)
    
    
    limiar_superior = confint[:, 1] - acf_x
    limiar_inferior = confint[:, 0] - acf_x

    lags_selecionados = []
    
    for i in range(1, max_lag+1):

        
        if acf_x[i] >= limiar_superior[i] or acf_x[i] <= limiar_inferior[i]:
            lags_selecionados.append(i-1)  #-1 por conta que o lag 1 em python é o 0
    
    #caso nenhum lag seja selecionado, essa atividade de seleção para o gridsearch encontrar a melhor combinação de lags
    if len(lags_selecionados)==0:


        print('NENHUM LAG POR ACF')
        lags_selecionados = [i for i in range(max_lag)]

    print('LAGS', lags_selecionados)

    #inverte o valor dos lags para usar na lista de dados
    lags_selecionados = [max_lag - (i+1) for i in lags_selecionados]



    return lags_selecionados

def split_serie_less_lags(series, perc_train, perc_val = 0): 
    import numpy as np   
      
    train_size = np.fix(len(series) *perc_train)
    train_size = train_size.astype(int)
    
    if perc_val > 0:
        
        val_size = np.fix(len(series) *perc_val).astype(int)
        
        x_train = series[0:train_size]
        x_val = series[train_size:train_size+val_size]        
        x_test = series[(train_size+val_size):-1]

        return x_train, x_test, x_val
        
    else:
        
                
        x_train = series[0:train_size+1]
        x_test = series[train_size:-1]
        

        return x_train, x_test
		
def select_validation_sample(serie, perc_val):
    tam = len(serie)
    val_size = np.fix(tam *perc_val).astype(int)
    return serie[0:tam-val_size,:],  serie[tam-val_size:-1,:]



def split_train_val_test(serie, p_tr, p_v1, p_v2):
    '''
    two validations sample
    '''
    tam_serie =  len(serie)
    #print(tam_serie)
    tam_train = round(p_tr * tam_serie)
    tam_val1 = round(p_v1 * tam_serie)
    tam_val2 = round(p_v2 * tam_serie)
    #tam_test = tam_serie - (tam_train +  tam_val1 + tam_val2 )



    return  serie[0:tam_train] , serie[tam_train:tam_train+tam_val1] , serie[tam_train+tam_val1:tam_train+tam_val1+tam_val2] , serie[tam_train+tam_val1+tam_val2: ]


