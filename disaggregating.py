import random
import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from scipy.stats import pearsonr
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import gdal
import matplotlib.pyplot as plt
import time
from train_no_water import sample_data,get_score
from sklearn import linear_model
from sklearn.metrics import r2_score
labels = np.array([1,2,4])
label_name = dict(zip(labels,['Cultivated land','Forest and grassland','water', 'built up land']))

def reclass(x):
    if x == 10:
        return 1 ## cultivated land
    elif x <= 50:
        return 2 ## forest and grassland
    elif x == 60:
        return 3 ## water 
    elif x == 80:
        return 4 ## built-up land
    else:
        return 0 ## mask value

import matplotlib
matplotlib.rc('font',size=6)

if __name__ == '__main__':
    ## train_im has 8bands 1-6 bands come from the band  2-7 of landsat 8 OLI,
    ## band 7 comes from the ASTEM dem which has an original resoluation of 90m
    ## band 8 comes from the VIIRS DNB which has an original resoluation of about 300m
    ## the label comes form the comparison result of land cover data 2010 from global30 project which has a high resoluation of 30m and high accuracy, and the lucc data from resdc which has a lower resolution of 100m and lower accuracy.
    ## weight model trainning
    print('disaggregating')
    
    res = pd.read_csv('validate_5_11_v1/county_sta14.csv').set_index('index')
    res['pop_pre'] = 0
    res['gdp_pre'] = 0
    ##res = res.loc[res.index[:-1]]
    dnb = gdal.Open('svdnb_30.tif').ReadAsArray()
    xzq = gdal.Open('xzq_beijing.tif').ReadAsArray()
    xzq2 = np.array([xzq,xzq])
    lucc = np.fromfile('validate_5_11_v1/predict_lucc.txt',sep = ',').reshape(xzq.shape)
    trans = gdal.Open('xzq_beijing.tif').GetGeoTransform()
    print(trans)
    weights = np.fromfile('validate_5_11_v1/predict_weight14.txt',sep=',').reshape((2,xzq.shape[0],xzq.shape[1]))
    weights = weights*(weights>0)
    for i in range(2):
        weights[i] = weights[i]*((lucc==1)+(lucc==4))
    t_weights = [np.sum(x) for x in weights]
    ## weights[0]: pop; weights[1]:GDP
    ##lucc = np.fromfile('validate_5_11_v1/predict_lucc.txt',sep=',').reshape(xzq.shape)


    ## validating at county level
    figure = plt.figure(figsize=(6,3),dpi=300)
    results = np.zeros(weights.shape)
    
    for i in range(2):
        results[i] = weights[i]/t_weights[i]*[res['pop_2015'].sum(),res['GDP_2015'].sum()][i] 
    for idx in res.index:
        res['pop_pre'][idx] = np.sum((xzq==idx)*(results[0]))
        res['gdp_pre'][idx] = np.sum((xzq==idx)*(results[1]))
        print(res.loc[idx,['pop_pre','gdp_pre']])

## plotting
    xticks = [np.arange(-2,7,1),np.arange(-2,11,2)]
    for i in range(2):
        model = linear_model.LinearRegression()
        ax = plt.subplot(1,2,i+1)
        y1 = np.log2(res[['pop_2015','GDP_2015'][i]]/res['count']*10000)
        y2 = np.log2(res[['pop_pre','gdp_pre'][i]]/res['count']*10000)
        plt.xlim(y1.min(),y1.max())
        plt.ylim(y1.min(),y1.max())
        plt.scatter(y1,y2,1.5,marker='o')
        plt.plot(y1,y1,'-  -', label= 'y = x',linewidth=1)
        model.fit(y1.reshape(-1,1),y2)
        r = r2_score(y1,y2) 
        plt.plot(y1,model.predict(y1.reshape(-1,1)),linewidth=1,
                 label= 'y = {0}*x + {1}\n $R^2$ = {2}'.format(round(model.coef_[0],4),
                                                               round(model.intercept_,4),round(r,4)))        
        plt.xlabel(['Population density of statistical data (people per grid)','GDP density of statistical data($10^4$ yuan per grid)'][i])
        plt.ylabel('Estimated '+['population density (people per grid)','GDP density ($10^4$ per grid)'][i])
        plt.xticks(xticks[i],[2**float(x) for x in xticks[i]])
        plt.yticks(xticks[i],[2**float(x) for x in xticks[i]])
        plt.title(['A: population density','B: GDP density'][i])
        plt.legend()
    plt.tight_layout()
    plt.savefig('analysis/figure_4.png')
    plt.show()

        


"""        
    print(pearsonr(res['pop_2015']/res['count'],res['pop_pre']/res['count']))
    print(pearsonr(res['GDP_2015']/res['count'],res['gdp_pre']/res['count']))
    

    
    for idx in res.index:
        for i in range(2):
            aa = (weights[i]*(xzq==idx))/np.sum((xzq==idx)*weights[i])*res[['pop_2015','GDP_2015'][i]][idx] 
            results[i] += aa
            res[['pop_pre','gdp_pre'][i]][idx] = np.sum(aa)
            print(res.loc[idx,['pop_pre','gdp_pre']])
        print(pearsonr(res['pop_2015'],res['pop_pre']))
        print(pearsonr(res['GDP_2015'],res['gdp_pre']))

    xticks = np.arange(115.5,117.6,0.5)
    yticks = np.arange(39.5,41.1,0.5)

    plt.subplot(2,2,1)
    plt.imshow(np.ma.masked_array(weights[0],xzq>100),cmap='Paired')
    plt.xticks([(x-trans[0])/trans[1] for x in xticks],[str(x)+'E' for x in xticks])
    plt.yticks([(x-trans[3])/trans[5] for x in yticks],[str(x)+'N' for x in yticks])
    plt.title('A: weights of population')
    plt.colorbar()

    plt.subplot(2,2,2)    
    plt.imshow(np.ma.masked_array(weights[1],xzq>100),cmap='Paired')
    plt.xticks([(x-trans[0])/trans[1] for x in xticks],[str(x)+'E' for x in xticks])
    plt.yticks([(x-trans[3])/trans[5] for x in yticks],[str(x)+'N' for x in yticks])
    plt.title('B: weights of GDP')
    plt.colorbar()
    
    plt.subplot(2,2,3) 
    plt.imshow(np.ma.masked_array(results[0]*10000,xzq>100),cmap='Paired')
    plt.xticks([(x-trans[0])/trans[1] for x in xticks],[str(x)+'E' for x in xticks])
    plt.yticks([(x-trans[3])/trans[5] for x in yticks],[str(x)+'N' for x in yticks])
    plt.title('C: population density (people per grid)')
    plt.colorbar()
    
    plt.subplot(2,2,4) 
    plt.imshow(np.ma.masked_array(results[1]*10000,xzq>100),cmap='Paired')
    plt.xticks([(x-trans[0])/trans[1] for x in xticks],[str(x)+'E' for x in xticks])
    plt.yticks([(x-trans[3])/trans[5] for x in yticks],[str(x)+'N' for x in yticks])
    plt.title('D: GDP density ($10^4$ yuan per grid)')
    plt.colorbar()

    plt.savefig('analysis/figure_5.png',dpi=320)

    plt.show()
"""    
