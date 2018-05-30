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

    

if __name__ == '__main__':
    ## train_im has 8bands 1-6 bands come from the band  2-7 of landsat 8 OLI,
    ## band 7 comes from the ASTEM dem which has an original resoluation of 90m
    ## band 8 comes from the VIIRS DNB which has an original resoluation of about 300m
    ## the label comes form the comparison result of land cover data 2010 from global30 project which has a high resoluation of 30m and high accuracy, and the lucc data from resdc which has a lower resolution of 100m and lower accuracy.
    ## weight model trainning
    print('aab')
    res = pd.read_csv('validate_5_11_v1/county_sta124.csv').set_index('index')
    ##res = res.loc[res.index[:-1]]
    dnb = gdal.Open('svdnb_30.tif').ReadAsArray()
    xzq = gdal.Open('xzq_beijing.tif').ReadAsArray()
    lucc = np.fromfile('validate_5_11_v1/predict_lucc.txt',sep=',').reshape(xzq.shape)
    res['dnb'] = 0.0
    for idx in res.index: res['dnb'][idx] = np.sum((xzq==idx)*(lucc!=3)*dnb[:xzq.shape[0]])
    res.to_csv('validate_5_11_v1/county_sta124.csv')
    columns = ['f_'+str(x) for x in range(256)]    
    res['pop_a'] = res['pop_2015']/res['count']*10000
    res['GDP_a'] = res['GDP_2015']/res['count']*10000
    for col in columns: res[col] = res[col]/res['count']
    res['dnb'] = res['dnb']/res['count']
    X1 = res['dnb'].reshape((-1,1))
    X2 = res[columns]
    X3 = res[columns+['dnb']]
    X = [X1,X2,X3]
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import linear_model
    from sklearn.cross_decomposition import PLSRegression

    ## fitting population
    figure = plt.figure(figsize=(10,5))
    y = res['pop_a']
    show_x = [0,1,5,10,30]
    t_pop = []
    n_pop = []
    yy = np.zeros((3,len(show_x)-1))
    ax1 = plt.subplot(1,2,1)
    ax2 = ax1.twinx()
    for i in range(3):
        ##model = RandomForestRegressor(10,max_depth=3,random_state=0)
        if X[i].shape[-1] > 1:
            model = linear_model.BayesianRidge()
        else:
            model = linear_model.LinearRegression()
        y_pre = cross_val_predict(model,X[i],y,cv=X[i].shape[0])
        y_pre = y_pre.reshape(-1)
        print(pearsonr(y,y_pre))
        
        for k in range(1,len(show_x)):
            RMSE = np.mean([(y_pre[j]-y[y.index[j]])**2 for j in range(len(y)) if y[y.index[j]] > show_x[k-1] and y[y.index[j]] <= show_x[k]])**0.5
            NRMSE = RMSE/np.mean([x for x in y if x>show_x[k-1] and x<show_x[k]])
            yy[i,k-1] = NRMSE
            if i == 0:
                pop = np.sum([res['GDP_2015'][x] for x in y.index if y[x]>show_x[k-1] and y[x]<=show_x[k]])
                n = np.count_nonzero([res['GDP_2015'][x] for x in y.index if y[x]>show_x[k-1] and y[x]<=show_x[k]])
                t_pop.append(pop)
                n_pop.append(n)
    
    ax1.bar(range(1,len(show_x)),t_pop,0.8,facecolor='0.5')
    for i in range(1,len(show_x)):
        ax1.text(i,t_pop[i-1]/4,'n = {0}'.format(n_pop[i-1]),ha='center')


    ax1.set_ylabel('Population distribution ($10^4$ people)')
    for i in range(3):
        ax2.plot(range(1,len(show_x)),yy[i],'*-', label = ['DNB','CNN','CNN+DNB'][i])
    ax2.set_ylabel('NRMSE')
    ax1.set_xlabel('Interval of population density (people per grid)')
    plt.legend()
    plt.xticks(range(1,len(show_x)),['{0}-{1}'.format(show_x[k-1],show_x[k]) for k in range(1,len(show_x))])
    ##plt.title('population')
    ##plt.savefig('analysis/figure3_pop.png')

    ## fitting GDP
    y = res['GDP_a']
    show_x = [0,10,100,500,1000]
    t_pop = []
    n_pop = []
    yy = np.zeros((3,len(show_x)-1))
    ax1 = plt.subplot(1,2,2)
    ax2 = ax1.twinx()
    for i in range(3):
        ##model = RandomForestRegressor(10,max_depth=3,random_state=0)
        if X[i].shape[-1] > 1:
            model = linear_model.BayesianRidge()
        else:
            model = linear_model.LinearRegression()
        y_pre = cross_val_predict(model,X[i],y,cv=X[i].shape[0])
        y_pre = y_pre.reshape(-1)
        print(pearsonr(y,y_pre))
        
        for k in range(1,len(show_x)):
            RMSE = np.mean([(y_pre[j]-y[y.index[j]])**2 for j in range(len(y)) if y[y.index[j]] > show_x[k-1] and y[y.index[j]] <= show_x[k]])**0.5
            NRMSE = RMSE/np.mean([x for x in y if x>show_x[k-1] and x<show_x[k]])
            yy[i,k-1] = NRMSE
            if i == 0:
                pop = np.sum([res['GDP_2015'][x] for x in y.index if y[x]>show_x[k-1] and y[x]<=show_x[k]])
                n = np.count_nonzero([res['GDP_2015'][x] for x in y.index if y[x]>show_x[k-1] and y[x]<=show_x[k]])
                t_pop.append(pop)
                n_pop.append(n)
    
    ax1.bar(range(1,len(show_x)),t_pop,0.8,facecolor='0.5')
    for i in range(1,len(show_x)):
        ax1.text(i,t_pop[i-1]/2,'n = {0}'.format(n_pop[i-1]),ha='center')
    
    ax1.set_ylabel('GDP distribution ($10^8$ yuan)')
    for i in range(3):
        ax2.plot(range(1,len(show_x)),yy[i],'*-', label = ['DNB','CNN','CNN+DNB'][i])
    ax2.set_ylabel('NRMSE')
    ax1.set_xlabel('Interval of GDP density ($10^4$ yuan per grid)')
    plt.legend()
    plt.xticks(range(1,len(show_x)),['{0}-{1}'.format(show_x[k-1],show_x[k]) for k in range(1,len(show_x))])
    ##plt.title('GDP')
    plt.tight_layout()
    plt.savefig('analysis/figure3_gdp.png')
    plt.show()

    if input('coutinue predict weight maps (y/n)') == 'n':
        sys.exit('stop now')
        
    


    ## create weight arrays(need to to finished)
    model_pop = linear_model.BayesianRidge()
    model_pop.fit(X2,res['pop_a'])
    print(model_pop.intercept_)
    model_gdp = linear_model.BayesianRidge()
    model_gdp.fit(X2,res['GDP_a'])
    print(model_gdp.intercept_)
    im = 'landsat_8.tif'
    res = pd.DataFrame(np.zeros((3,3)),index = labels,columns=['p_'+str(l) for l in labels])
    dnb = gdal.Open('SVDNB_30.tif').ReadAsArray()
    dem = gdal.Open('dem_30.tif').ReadAsArray()
    dnb = (dnb-dnb.min())/(dnb.max()-dnb.min()) 
    dem = (dem-dem.min())/(dem.max()-dem.min())
    xzq = gdal.Open('XZQ_beijing.tif').ReadAsArray()
    lc_array = np.fromfile('validate_5_11_v1/predict_lucc.txt',sep=',').reshape(xzq.shape)
    im_data = gdal.Open(im)
    batch_size = 1000
    print('loading data',time.ctime()) 
    im_trans = im_data.GetGeoTransform()
    im_data = im_data.ReadAsArray()[1:-1]
    im_array = im_data[:,:-1,:-1]
    im_array = im_array.astype('float')
    
    lc_2010 = gdal.Open('beijing_global30r.tif').ReadAsArray()[:lc_array.shape[0],:lc_array.shape[1]]
##    for i in range(lc_array.shape[0]):
##        for j in range(lc_array.shape[1]):
##            lc_array[i,j] = reclass(lc_array[i,j])
    print(im_array.shape)
    for i in range(im_array.shape[0]):
        im_array[i] = (im_array[i]-im_array[i].min())/(im_array[i].max()-im_array[i].min())
    im_array = np.append(im_array,[dem[:im_array.shape[1],:im_array.shape[2]],dnb[:im_array.shape[1],:im_array.shape[2]]],axis=0)    
    print(im_array.shape)
    ## seperate training and validating data

    saver = tf.train.import_meta_graph('4_29_home_mac_no_water/CNN_v2.meta')
    sess = tf.Session()
    ##res = input('restore (r) or begin new (n)')
    restore_file = '5_7_no_water/.' 
    saver.restore(sess,tf.train.latest_checkpoint(restore_file))
    y_pre = sess.graph.get_tensor_by_name('y_pre:0')
    h_fc1 = sess.graph.get_tensor_by_name('h_fc1:0')
    xs = sess.graph.get_tensor_by_name('xs:0')
    keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
    out_file = 'validate_5_11_v1/'
    if os.path.exists(out_file) == False:
        os.makedirs(out_file)
    predict = np.zeros((2,xzq.shape[0],xzq.shape[1]))
    extent = [4700,4800,3600,3700]
    extent = [0,im_array.shape[1],0,im_array.shape[2]]
    w = 9 
    for r in range(extent[0],extent[1]):
        c_list = [x for x in range(extent[2],extent[3]) if xzq[r,x] != 255]
        c_list = [x for x in c_list if lc_array[r,x] != 2]
        c_list = [x for x in c_list if lc_array[r,x] != 3]
        y_p = np.zeros((2,len(c_list)))
        for i in range(len(c_list)//batch_size+1*(len(c_list)%batch_size!=0)):
            sta,end = i*batch_size, min(i*batch_size+batch_size,len(c_list))
            bx = np.zeros((end-sta,w,w,8,1))
            for j in range(end-sta):
                c = c_list[sta+j]
                for b in range(8):
                    bx[j,:,:,b,0] = im_array[b,r-w//2:r+w//2+1,c-w//2:c+w//2+1]
            xx = sess.run(h_fc1,feed_dict={xs:bx,keep_prob:1.0})
            y_p[0,sta:end] = model_pop.predict(xx).reshape(-1)
            y_p[1,sta:end] = model_gdp.predict(xx).reshape(-1)
        predict[:,r,c_list] = y_p
        print(time.ctime(),'finished predicting of row: ',r)
    ##predict = (predict*(lc_array==0) + lc_array*(lc_array!=0))*(xzq>0)*(xzq<20)     
    predict.tofile(out_file+'predict_weight14.txt',sep=',')


    



    
    
