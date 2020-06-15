"""
Created on May 1st, 2020
Train U-Net + sampling pattern
@author: haggarwal
"""

import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime


import misc as sf #sf stands for supporting functions
import model as mm



config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.reset_default_graph()

#%
lam=.05
K=1
epochs=100
nSave=100
acc=6
sigma=.01
batchSz=1
lr=1e-3
#Please download the dataset from the google drive link and put the path here
dataset_filename='/Users/haggarwal/datasets/trndata_jmodl.npz'
#%% get the dataset

tmp = np.load(dataset_filename, mmap_mode='r')
trnOrg=tmp['trnOrg']
trnCsm=tmp['trnCsm']
nImg=trnOrg.shape[0]

M,N=trnOrg.shape[-2:]
initx=np.load('initmask'+str(acc)+'.npz')['kh']
inity=np.load('initmask'+str(acc)+'.npz')['kv']
initx=np.where(initx==True)[0][:,None].astype(np.float32)/M
inity=np.where(inity==True)[0][:,None].astype(np.float32)/N

#%%Generate a meaningful filename to save the trainined models for testing
print ('*************************************************')
start_time=time.time()
saveDir='trained_models/'
cwd=os.getcwd()
directory=saveDir+datetime.now().strftime("%d%b_%I%M%S%P_")+ \
 str(acc)+'acc_'+  str(nImg)+'I_'+str(epochs)+'ep_' +str(K)+'K'

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName= directory+'/model'

#%% save testing model first.

tf.reset_default_graph()
orgT=tf.placeholder(dtype=tf.complex64,shape=(None,None,None),name='org')
csmT = tf.placeholder(tf.complex64,shape=(None,None,None,None),name='csm')

atbT,predT=mm.makeModel(orgT,csmT,initx,inity,M,N,sigma,lam,K,False)
atbT =tf.identity(atbT, name='atb')
predT=tf.identity(predT,name='predTst')

sessFileNameTst=directory+'/modelTst'

saver=tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    savedFile=saver.save(sess, sessFileNameTst,latest_filename='checkpointTst')
print ('testing model saved:' +savedFile)

#%% some tensorflow code for dataset input
tf.reset_default_graph()
orgP =tf.placeholder(dtype=tf.complex64,shape=(None,None,None),name='org')
csmP =tf.placeholder(tf.complex64,shape=(None,None,None,None),name='csm')

trnData = tf.data.Dataset.from_tensor_slices( (orgP,csmP))
trnData = trnData.cache()
trnData=trnData.repeat(count=epochs)
trnData = trnData.shuffle(buffer_size=50)
trnData=trnData.batch(batchSz)
iterator=trnData.make_initializable_iterator()
orgT,csmT = iterator.get_next()
#%% make training model
_,predT=mm.makeModel(orgT,csmT,initx,inity,M,N,sigma,lam,K,True)
predT=sf.tf_r2c(predT)
tmp1=tf.abs(predT-orgT)
loss= tf.reduce_mean(tf.pow(tmp1,2))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
opToRun=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
opToRun=tf.group([update_ops,opToRun])
tSmry=tf.summary.scalar('tloss', loss)

#%% perform the training
nBatch= int(np.floor(np.float32(nImg)/batchSz))
nSteps= nBatch*epochs
ep=0
err=np.zeros(epochs)

print ('training started at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
print ('parameters are: Epochs:',epochs,' BS:',batchSz,'nSteps:',nSteps,'nSamples:',nImg)

saver = tf.train.Saver(max_to_keep=100)
totalLoss,ep=[],0
lossT = tf.placeholder(tf.float32)
lossSumT = tf.summary.scalar("TrnLoss", lossT)
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    feedDict={orgP:trnOrg,csmP:trnCsm}
    sess.run(iterator.initializer,feed_dict=feedDict)
    savedFile=saver.save(sess, sessFileName)
    print("Model meta graph saved in::%s" % savedFile)

    writer = tf.summary.FileWriter(directory, sess.graph)
    for step in tqdm(range(nSteps)):
        try:
            tmp,_=sess.run([loss,opToRun])

            totalLoss.append(tmp)
            if np.remainder(step+1,nBatch)==0:
                ep=ep+1
                avgTrnLoss=np.mean(totalLoss)
                lossSum=sess.run(lossSumT,feed_dict={lossT:avgTrnLoss})
                writer.add_summary(lossSum,ep)
                if ep % nSave==0:
                    savedfile=saver.save(sess, sessFileName,global_step=ep,
                                         write_meta_graph=True)

                totalLoss=[] #after each epoch empty the list of total loos
        except tf.errors.OutOfRangeError:
            break
    savedfile=saver.save(sess, sessFileName,global_step=ep,write_meta_graph=True)
    writer.close()

end_time = time.time()
print ('Trianing completed in minutes ', ((end_time - start_time) / 60))
print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
print ('*************************************************')


