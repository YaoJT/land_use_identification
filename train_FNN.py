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


labels = np.array([1,2,3,4])
label_name = dict(zip(labels,['Cultivated land','Forest and grassland','water', 'built up land']))
def sample_data(train_file,points,sample_size):
    global labels
    bx,by=[],[]
    for l in sample_size:
        add_data = points[points['labels']==l].sample(sample_size[l])
        add_x = [gdal.Open(os.path.join(train_file,x)).ReadAsArray() for x in add_data['im_f']]
        add_x = [[x/x.max() for x in y] for y in add_x]
        bx += add_x
        by += [(labels==x)+0 for x in add_data['labels']]
        
    return np.array(bx),np.array(by)
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
def sample_data(im_array,lc_array,batch_arnray,batch_num = 1,n=24,ny=12):
    global labels
    batch_size = np.sum(batch_array == batch_num)
    X = np.zeros((batch_size,2*n,2*n,im_array.shape[0]))
    y = np.zeros((batch_size,len(labels)))
    indexes = np.argwhere(batch_array==batch_num)
    for i in range(indexes.shape[0]):
        if indexes[i].min()>50 and indexes[i,0]<im_array.shape[1]-50 and indexes[i,1]<im_array.shape[2]-50:
            X[i] = im_array[:,indexes[i,0]-n:indexes[i,0]+n,indexes[i,1]-n:indexes[i,1]+n]
            y[i] = np.array(labels) == lc_array[indexes[i,0],indexes[i,1]]
    return X,y    

def get_error(errors,y):
    global labels
    out_err = {}
    for la in range(len(labels)):
        out_err[label_name[labels[la]]] = np.mean([errors[i] for i in range(len(errors)) if y[i][la] == 1])
    out_err['mean'] = np.mean(errors)
    return out_err

def get_score(X,y):
    y_p = sess.run(y_pre,feed_dict={xs:X,keep_prob:1})
    y_p = np.array([list(m).index(np.max(m)) for m in y_p])
    y = np.array([list(m).index(np.max(m)) for m in y])
    scores = {}
    for la in range(len(labels)):
        TP = np.sum((y==y_p)*(y==la))/np.sum(y==la) if np.sum(y==la) > 0 else 0
        PT = np.sum((y==y_p)*(y_p==la))/np.sum(y_p==la) if np.sum(y_p==la) > 0 else 0
        scores[labels[la]] = [TP,PT,2*TP*PT/(TP+PT) if TP+PT >0 else 0]
    scores['mean'] = np.mean([x[-1] for x in scores.values()])
    return scores

def validate():

    return 0

def predict():
    return 0

    

if __name__ == '__main__':
    ## train_im has 8bands 1-6 bands come from the band  2-7 of landsat 8 OLI,
    ## band 7 comes from the ASTEM dem which has an original resoluation of 90m
    ## band 8 comes from the VIIRS DNB which has an original resoluation of about 300m
    ## the label comes form the comparison result of land cover data 2010 from global30 project which has a high resoluation of 30m and high accuracy, and the lucc data from resdc which has a lower resolution of 100m and lower accuracy.
    print('4_23')
    im = 'c:/PHDthesis/data/landsat_8.tif'
    dnb = gdal.Open('c:/PHDthesis/data/SVDNB_30.tif').ReadAsArray()
    dem = gdal.Open('c:/PHDthesis/data/dem_30.tif').ReadAsArray()
    dnb = (dnb-dnb.min())/(dnb.max()-dnb.min()) 
    dem = (dem-dem.min())/(dem.max()-dem.min())
    lc = 'c:/PHDthesis/data/unchanged_10_15.tif'
    im_data = gdal.Open(im)
    batch_size = 100 
    batch_num = 4000 
    train_ratio = 0.9
    print('loading data',time.ctime()) 
    lc_data = gdal.Open(lc)
    im_trans = im_data.GetGeoTransform()
    lc_trans = lc_data.GetGeoTransform()
    im_array = im_data.ReadAsArray()[1:-1]
    im_array = im_array[:,:-1,:-1]
    print(im_array.shape)
    for i in range(im_array.shape[0]):
        im_array[i] = (im_array[i]-im_array.min())/(im_array.max()-im_array.min())
    im_array = np.append(im_array,[dem[:im_array.shape[1],:im_array.shape[2]],dnb[:im_array.shape[1],:im_array.shape[2]]],axis=0)    
    print(im_array.shape)
    lc_array = lc_data.ReadAsArray()[:im_array.shape[1],:im_array.shape[2]]
    print(lc_array.shape)
    ## seperate training and validating data

    print('getting batch array',time.ctime())
    batch_array = np.zeros(lc_array.shape)
    for lc in labels:
        r = batch_size*batch_num/len(labels)/np.sum(lc_array==lc)
        batch_array = batch_array + ((lc_array==lc)*(np.random.random(lc_array.shape)<r))
    batch_array = batch_array*np.random.randint(1,batch_num+1,batch_array.shape)
    batch_array.tofile('batch_array.txt',sep=',')

##    print('batch array reading finished',time.ctime())
##    for la in labels:
##        print(la,np.sum((batch_array>0)*(lc_array==la)))
    print('building graph', time.ctime())
    X,y = sample_data(im_array,lc_array,batch_array,1)
    print(X.shape,y.shape)
#    plt.imshow(batch_array)
#    plt.show()
                
    im_shape = (48,48)
    npp_class = 4
    ## define the graph of tf model
    xs = tf.placeholder(tf.float32, shape = (None,8,im_shape[0],im_shape[1],1), name = 'xs')
    ys = tf.placeholder(tf.float32, shape = [None,npp_class], name = 'ys')
    keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')


    ## conv1 layer
    W_conv1 = tf.Variable(tf.truncated_normal([2,2,2,1,128],stddev=0.1))
    B_conv1 = tf.Variable(tf.constant(0.1,shape=[128]))
    h_conv1 = tf.nn.conv3d(xs,W_conv1,[1,1,1,1,1],'SAME',name='h_conv1')
    h_relu1 = tf.nn.relu(h_conv1+B_conv1, name = 'h_relu1')
    h_pool1 = tf.nn.max_pool3d(h_conv1,[1,2,2,2,1],[1,2,2,2,1],'SAME', name = 'h_pool1') ## output size -1,4*24*24,64 
    ## conv2 
    W_conv2 = tf.Variable(tf.truncated_normal([2,2,2,128,128],stddev = 0.1))
    B_conv2 = tf.Variable(tf.constant(0.1,shape=[128]))
    h_conv2 = tf.nn.conv3d(h_pool1,W_conv2,[1,1,1,1,1],'SAME',name = 'h_conv2')
    h_relu2 = tf.nn.relu(h_conv2+B_conv2, name = 'h_relu2')
    h_pool2 = tf.nn.max_pool3d(h_relu2,[1,2,2,2,1],[1,2,2,2,1],'SAME', name = 'h_pool2')   ## output size -1, 2*12*12,128
    ## conv3 
    W_conv3 = tf.Variable(tf.truncated_normal([2,2,2,128,128],stddev = 0.1))
    B_conv3 = tf.Variable(tf.constant(0.1,shape=[128]))
    h_conv3 = tf.nn.conv3d(h_pool2,W_conv3,[1,1,1,1,1],'SAME',name = 'h_conv3')
    h_relu3 = tf.nn.relu(h_conv3+B_conv3, name = 'h_relu3')
    h_pool3 = tf.nn.max_pool3d(h_relu3,[1,2,2,2,1],[1,2,2,2,1],'SAME', name = 'h_pool3')   ## output size -1, 1*6*6,128
    #
    ## fc1
    W_fc1 = tf.Variable(tf.truncated_normal([36*128,256],stddev=0.1))
    B_fc1 = tf.Variable(tf.constant(0.1,shape=[256]))
    h_pool3_flat = tf.reshape(h_pool3,[-1,36*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1) + B_fc1, name = 'h_fc1')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name = 'h_fc1_drop')
    ## fc2
    W_fc2 = tf.Variable(tf.truncated_normal([256,512],stddev=0.1))
    B_fc2 = tf.Variable(tf.constant(0.1,shape=[512]))
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+B_fc2, name = 'h_fc2')
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob, name = 'h_fc2_drop')
    ## fc3
    W_fc3 = tf.Variable(tf.truncated_normal([512,1024],stddev=0.1))
    B_fc3 = tf.Variable(tf.constant(0.1,shape=[1024]))
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop,W_fc3)+B_fc3, name = 'h_fc3')
    h_fc3_drop = tf.nn.dropout(h_fc3,keep_prob, name = 'h_fc3_drop')
    ## last layer
    W_fc = tf.Variable(tf.truncated_normal([1024,4],stddev=0.1))
    B_fc = tf.Variable(tf.constant(0.1,shape=[4]))
    y_pre = tf.nn.softmax(tf.matmul(h_fc3,W_fc)+B_fc,name='y_pre')
    ## error and train
    error = tf.nn.softmax_cross_entropy_with_logits(labels = ys,logits=tf.matmul(h_fc3,W_fc)+B_fc,name='error')
    train_step = tf.train.AdamOptimizer(0.0001).minimize(tf.reduce_mean(error))
    tf.add_to_collection('train_step',train_step)
    ## start training
    saver = tf.train.Saver()
    sess = tf.Session()
    ##res = input('restore (r) or begin new (n)')
##    if len(sys.argv) == 1:
    init = tf.global_variables_initializer()
    sess.run(init)
##    elif len(sys):
##        restore_file = input('input restore file: ')
##        saver.restore(sess,tf.train.latest_checkpoint(restore_file))

    out_file = '4_23/'
    if os.path.exists(out_file) == False:
        os.makedirs(out_file)
    scores = []
##    plt.ion()
    for e in range(100):
        for b in range(1,int(batch_num*0.9)+1):
            i = e * batch_num + b    
            batch_x, batch_y = sample_data(im_array,lc_array,batch_array,b)
            batch_x = batch_x.reshape(list(batch_x.shape)+[1])
            if i%5 == 0 or i == 1:
                scores.append(get_score(batch_x,batch_y))
                print(time.ctime())
                print('epoch:{0}-{1}/{2},scores:{3}'.format(e,b,batch_num,scores[-1]))   
                np.array(scores).tofile(out_file+'score.txt',',')
                if i%100 == 0 or i == 1:
                    plt.cla()
                    plt.plot([x['mean'] for x in scores],lw=2,label = 'mean')
                    for la in labels:
                        plt.plot([x[la][-1] for x in scores],lw=0.5,label=label_name[la])
                    plt.legend()
                    plt.ylim(0,1)
                    plt.savefig(out_file+'score.png')
                    for k in range(10):
                        try:
                            saver.save(sess,out_file+'CNN_v2',i)
                            print('model saved')
                            break
                        except:
                            print('save again')
                            time.sleep(10)
            sess.run(train_step,feed_dict={xs:batch_x,ys:batch_y,keep_prob:0.7})
        ##        ## validating:
##        v_y = np.zeros(np.sum(batch_array>(batch_num*0.9)))
##        v_y_pre = np.zeros(np.sum(batch_array>(batch_num*0.9)))
##        v = 0
##        for vb in range(int(batch_num*0.9)+1,batch_num+1):
##            indexes = np.argwhere(batch_array == vb)
##            v_X = np.zeros(([len(indexes)]+list(im_array.shape)))
##            for idx in indexes:
            
            ##plt.cla()
                ##for k in range(len(labels)):
                ##    plt.plot([x[k] for x in errors],label=labels[k])
                ##plt.plot([np.mean(x) for x in errors],label='mean',lw=2)
                
    
    


    



    
    
