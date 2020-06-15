"""
Created on Tue Sep 17 10:35:59 2019

@author: haggarwal
"""
import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import misc as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.reset_default_graph()
#%%

directory='15Jun_023015pm_6acc_100I_100ep_1K'
modelDir='trained_models/'+directory
#%% get the dataset

tmp = np.load('tstdata_jmodl.npz', mmap_mode='r')
tstOrg=tmp['tstOrg']
tstCsm=tmp['tstCsm']
nImg=tstOrg.shape[0]
#%%

tstRec =np.zeros(tstOrg.shape+(2,),dtype=np.float32)
tstAtb =np.zeros_like(tstOrg)
loadChkPoint=tf.train.latest_checkpoint(modelDir)

with tf.Session(config=config) as sess:
    new_saver = tf.train.import_meta_graph(modelDir+'/modelTst.meta')
    new_saver.restore(sess, loadChkPoint)
    graph = tf.get_default_graph()
    orgT=graph.get_tensor_by_name('org:0')
    atbT=graph.get_tensor_by_name('atb:0')
    csmT=graph.get_tensor_by_name('csm:0')
    kxT =graph.get_tensor_by_name('kx:0')
    kyT =graph.get_tensor_by_name('ky:0')
    recT=graph.get_tensor_by_name('predTst:0')
    wts=sess.run(tf.global_variables())
    kx,ky=sess.run([kxT,kyT])
    for i in tqdm(range(nImg)):
        fd={orgT:tstOrg[[i]],csmT:tstCsm[[i]]}
        tstAtb[i],tstRec[i]=sess.run([atbT,recT],feed_dict=fd)

tstRec=sf.r2c(tstRec)

#%
fn= lambda x: sf.normalize01(np.abs(x))

normOrg=fn(tstOrg)
normAtb=fn(tstAtb)
normRec=fn(tstRec)

psnrAtb=sf.myPSNR(normOrg,normAtb).mean()
psnrRec=sf.myPSNR(normOrg,normRec).mean()

print ('  ' + 'Noisy ' + 'Reconst.')
print ('  {0:.2f} {1:.3f}'.format(psnrAtb,psnrRec))

#%%
def getMask(kx,ky):
    kxF=kx*255
    kyF=ky*231
    kxF=np.int32(kxF)
    kyF=np.int32(kyF)
    mx=np.zeros((256,1),dtype=np.bool)
    mx[kxF]=True
    my=np.zeros((232,1),dtype=np.bool)
    my[kyF]=True
    mask=mx@(my.T)
    return mask
mask=getMask(kx,ky)

#%% view reconstruction
plt.figure()
plt.subplot(1,4,1)
plt.imshow(normOrg[0],cmap='gray')
plt.title('Orginal Image')
plt.axis('off')
plt.subplot(1,4,2)
plt.imshow(mask,aspect='equal',cmap='gray')
plt.axis('off')
plt.title('Learned mask, 6-fold')
plt.subplot(1,4,3)
plt.imshow(normAtb[0],cmap='gray')
plt.title('Aliased image, PSNR='+str(round(psnrAtb,2))+'dB')
plt.axis('off')
plt.subplot(1,4,4)
plt.imshow(normRec[0],cmap='gray')
plt.axis('off')
plt.title('Reconstructed image PSNR='+str(round(psnrRec,2))+'dB')
plt.subplots_adjust(left=0, right=1, top=.93, bottom=0,wspace=0.01)
plt.show()


