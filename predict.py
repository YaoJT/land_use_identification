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
    print('5_9_validate')
    im = 'landsat_8.tif'
    res = pd.DataFrame(np.zeros((3,3)),index = labels,columns=['p_'+str(l) for l in labels])
    dnb = gdal.Open('SVDNB_30.tif').ReadAsArray()
    dem = gdal.Open('dem_30.tif').ReadAsArray()
    dnb = (dnb-dnb.min())/(dnb.max()-dnb.min()) 
    dem = (dem-dem.min())/(dem.max()-dem.min())
    xzq = gdal.Open('XZQ_beijing.tif').ReadAsArray()
    lc = 'lucc_2015.tif'
    im_data = gdal.Open(im)
    batch_size = 1000
    print('loading data',time.ctime()) 
    lc_data = gdal.Open(lc)
    im_trans = im_data.GetGeoTransform()
    lc_trans = lc_data.GetGeoTransform()
    im_data = im_data.ReadAsArray()[1:-1]
    im_array = im_data[:,:-1,:-1]
    im_array = im_array.astype('float')
    lc_array = lc_data.ReadAsArray()
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
    xs = sess.graph.get_tensor_by_name('xs:0')
    keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
    out_file = 'validate_5_11_v1/'
    if os.path.exists(out_file) == False:
        os.makedirs(out_file)
    predict = np.zeros(xzq.shape)
    extent = [4700,4800,3600,3700]
    extent = [0,im_array.shape[1],0,im_array.shape[2]]
    w = 9 
    for r in range(extent[0],extent[1]):
        c_list = [x for x in range(extent[2],extent[3]) if xzq[r,x] != 255]
        y_p = [0]*len(c_list)
        for i in range(len(c_list)//batch_size+1*(len(c_list)%batch_size!=0)):
            sta,end = i*batch_size, min(i*batch_size+batch_size,len(c_list))
            bx = np.zeros((end-sta,w,w,8,1))
            for j in range(end-sta):
                c = c_list[sta+j]
                for b in range(8):
                    bx[j,:,:,b,0] = im_array[b,r-w//2:r+w//2+1,c-w//2:c+w//2+1]
            xx = sess.run(y_pre,feed_dict={xs:bx,keep_prob:1.0})
            xx = sess.run(tf.argmax(xx,1))
            y_p[sta:end] = [labels[x] for x in xx]
        predict[r,c_list] = y_p
        print(time.ctime(),'finished predicting of row: ',r)
    ## validating results
    res['recall'] = 0.0
    for la in labels:
        for lap in labels:
            res['p_'+str(lap)][la] = np.sum((lc_array[extent[0]:extent[1],extent[2]:extent[3]]==la)*(predict[extent[0]:extent[1],extent[2]:extent[3]]==lap))
        res['recall'][la] = res['p_'+str(la)][la]/np.sum(res.loc[la])    
    print(res)
    res.to_csv(out_file+'validate.csv')
    ##predict = (predict*(lc_array==0) + lc_array*(lc_array!=0))*(xzq>0)*(xzq<20)     
    predict = predict*(lc_2010 != 60) + 3*(lc_2010==60)*(xzq>0)
    predict.tofile(out_file+'predict_lucc.txt',sep=',')
    ## plot remote imagery
    plt.subplot(2,2,1)
    im_show = np.zeros((extent[1]-extent[0],extent[3]-extent[2],3))
    im_show[:,:,0] = im_array[2,extent[0]:extent[1],extent[2]:extent[3]]
    im_show[:,:,1] = im_array[1,extent[0]:extent[1],extent[2]:extent[3]]
    im_show[:,:,2] = im_array[0,extent[0]:extent[1],extent[2]:extent[3]]
    plt.imshow(im_show/im_show.max())
    ## plot classification result of LCUU2015
    plt.subplot(2,2,2)
    lc_array = np.ma.masked_array(lc_array,lc_array<0)
    plt.imshow(lc_array[extent[0]:extent[1],extent[2]:extent[3]])
    ##plt.colorbar()
    plt.title('LUCC')
    ## plot classification result of our model
    plt.subplot(2,2,3)
    plt.imshow(lc_2010[extent[0]:extent[1],extent[2]:extent[3]])
    ##plt.colorbar()
    plt.title('lucc2010')
    ##plt.legend()
    plt.subplot(2,2,4)
    plt.imshow(predict[extent[0]:extent[1],extent[2]:extent[3]])
    plt.title('CNN')
    plt.show()
    


    



    
    
