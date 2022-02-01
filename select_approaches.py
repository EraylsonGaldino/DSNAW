import numpy as np

def selec_model_ola_erro(window_test_inst, past_data, ensemble, k, lags_acf):
    from scipy.spatial.distance import euclidean 

    #retorna o erro dos modelos
   
    #max_lag = len(lags_acf)
    x_data = past_data[:,lags_acf]
    y_data = past_data[:,-1]
    
    tam = len(x_data[0]) 
    
    dist = []
    for i in range(0,len(x_data)):
        d = euclidean(window_test_inst, x_data[i,:])
        #print(d)
        dist.append(d)
        
    indices_patterns = range(0, len(x_data))
    
    dist, indices_patterns = zip(*sorted(zip(dist, indices_patterns))) #returna tuplas ordenadas
    indices_patterns_l = list(indices_patterns)

    k_patterns_x = x_data[indices_patterns_l[0:k]]
    k_patterns_y = y_data[indices_patterns_l[0:k]]
  
    erros_modelos = []
    for i in range(0, len(ensemble)):
        model = ensemble[i]
        current_patterns = k_patterns_x
        
        
        prev = model.predict(current_patterns)
 
        er = 0
        #import pdb; pdb.set_trace()
        for j in range(0, len(prev)):
            er = er + np.absolute(k_patterns_y[j] - prev[j])



        erros_modelos.append(er)       
   
    
    return erros_modelos


def select_model_less(values_error):

    #seleciona o modelo com menor valor
    
    indices = range(0, len(values_error))
    
    valores_ordenados, indices = zip(*sorted(zip(values_error, indices))) #returna tuplas ordenadas
    
    return indices, indices[0]

def selec_model_best_before(previous_data, ensemble,lags_acf):

    error = []
    
    x_data = previous_data[:, lags_acf]
    y_data = previous_data[:,-1]
    
    
    for i in range(0, len(ensemble)):
        model = ensemble[i]
               
        prev = model.predict(x_data)
        er = 0
        
        for j in range(0, len(prev)):
            er = er + np.absolute(y_data[j] - prev[j])
        
        error.append(er)
    
              
    return error

def dsnaw(previous_data, ensemble,lags_acf, k = 10, m = 1):
    '''Seleciona utilização a região de competência formada por k janelas anteriores'''
    '''
    param:
        previous_data: all time windows previous the new instance
        ensemble: dict with m models
        k: size of the region of competence
        lags_acf: list of lags selected
        n: number of models to select 

    ''' 

    region_of_competence = previous_data[-k:]
    error_models = selec_model_best_before(region_of_competence, ensemble,lags_acf)
    ind_sel_bpm, ind_best_model = select_model_less(error_models)
    
    return  list(ind_sel_bpm[0:m])

def ds(instance, train_lags, ensemble, k, lags_acf, n = 1):
    '''Seleciona utilizando a região de competência por similaridade'''

    '''
    param: 
        instance: time windows in out-of-sample
        train_lags: sample of time windows in training sample
        ensemble: list with m models
        k: size of the region of competence
        lags_acf: list of lags selected
        n: number of models to select

    '''
    erros_ds = selec_model_ola_erro(instance, train_lags, ensemble, k, lags_acf)
    ind_sel_ds, modelo_ds = select_model_less(erros_ds)
    return ind_sel_ds[0:n]