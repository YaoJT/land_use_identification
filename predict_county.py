import random
import sys
from sklearn.metrics import accuracy_score
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
    print('5_13_validateaa')
    out_file = 'validate_5_11_v1/'
    fac_num = 256
    res = pd.read_csv(out_file+'county_sta124.csv').set_index('index')
    
    columns = ['f_'+str(x) for x in range(fac_num)]
    im = 'landsat_8.tif'
    dnb = gdal.Open('SVDNB_30.tif').ReadAsArray()
    dem = gdal.Open('dem_30.tif').ReadAsArray()
    dnb = (dnb-dnb.min())/(dnb.max()-dnb.min()) 
    dem = (dem-dem.min())/(dem.max()-dem.min())
    xzq = gdal.Open('XZQ_beijing.tif').ReadAsArray()
    lc = 'validate_5_11_v1/predict_lucc.txt'
    lc_2010 = 'beijing_global30r.tif'
    im_data = gdal.Open(im)
    batch_size = 1000
    print('loading data',time.ctime()) 
    lc_array = np.fromfile(lc,sep=',').reshape(xzq.shape) 
    im_trans = im_data.GetGeoTransform()
    im_data = im_data.ReadAsArray()[1:-1]
    im_array = im_data[:,:-1,:-1]
    im_array = im_array.astype('float')
    
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
    h_fc1 = sess.graph.get_tensor_by_name('h_fc1:0')
    xs = sess.graph.get_tensor_by_name('xs:0')
    keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
    if os.path.exists(out_file) == False:
        os.makedirs(out_file)
    extent = [0,xzq.shape[0],0,xzq.shape[1]]
    w = 9 
    
    
    for idx in res.index:
        res['count'][idx] = np.sum((xzq==idx))
        print(res.loc[idx])

        for r in range(extent[0],extent[1]):
            c_list = [x for x in range(extent[2],extent[3]) if xzq[r,x] == idx]
            c_list = [x for x in c_list if lc_array[r,x] == 1 or lc_array[r,x]==4]
            y_p = np.zeros((len(c_list),fac_num))
            for i in range(len(c_list)//batch_size+1*(len(c_list)%batch_size!=0)):
                sta,end = i*batch_size, min(i*batch_size+batch_size,len(c_list))
                bx = np.zeros((end-sta,w,w,8,1))
                for j in range(end-sta):
                    c = c_list[sta+j]
                    for b in range(8):
                        bx[j,:,:,b,0] = im_array[b,r-w//2:r+w//2+1,c-w//2:c+w//2+1]
                xx = sess.run(h_fc1,feed_dict={xs:bx,keep_prob:1.0})
                y_p[sta:end,:] = xx
            res.loc[idx,columns] += np.sum(y_p,axis=0)    
            print(time.ctime(), 'finished predicting of row: {0} of county {1}'.format(r,res['name'][idx]))
        res.to_csv(out_file+'county_sta14.csv')

    



    
    
