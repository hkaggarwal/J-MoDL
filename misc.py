"""
Created on Aug 6th, 2018

This file contains some supporting functions used during training and testing.

@author:Hemant
"""

import numpy as np
import tensorflow as tf
import mkl_fft

from skimage.metrics import structural_similarity


def normalize01(img):
    """
    Normalize the image between 0 and 1
    """
    shp=img.shape
    if np.ndim(img)>=3:
        nimg=np.prod(shp[0:-2])
    elif np.ndim(img)==2:
        nimg=1
    img=np.reshape(img,(nimg,shp[-2],shp[-1]))
    eps=1e-15
    img2=np.empty_like(img)
    for i in range(nimg):
        mx=img[i].max()
        mn=img[i].min()
        img2[i]= (img[i]-mn)/(mx-mn+eps)

    img2=np.reshape(img2,shp)
    return img2
#%%

def fft2c(img):
    """
    it works on last two dimensions. takes image domain data and do the
    fft2 to return kspace data
    """
    shp=img.shape
    nimg=int(np.prod(shp[0:-2]))
    scale=1/np.sqrt(np.prod(shp[-2:]))
    img=np.reshape(img,(nimg,shp[-2],shp[-1]))

    tmp=np.empty_like(img,dtype=np.complex64)
    for i in range(nimg):
        #tmp[i]=scale*np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img[i])))
        tmp[i]=scale*np.fft.fftshift(mkl_fft.fft2(np.fft.ifftshift(img[i])))

    kspace=np.reshape(tmp,shp)
    return kspace


def ifft2c(kspace):
    """
    it works on last two dimensions. takes image domain data and do the
    fft2 to return kspace data
    """
    shp=kspace.shape
    scale=np.sqrt(np.prod(shp[-2:]))
    nimg=int(np.prod(shp[0:-2]))

    kspace=np.reshape(kspace,(nimg,shp[-2],shp[-1]))

    tmp=np.empty_like(kspace)
    for i in range(nimg):
        #tmp[i]=scale*np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace[i])))
        tmp[i]=scale*np.fft.fftshift(mkl_fft.ifft2(np.fft.ifftshift(kspace[i])))

    img=np.reshape(tmp,shp)
    return img


def tf_fft2c(kspace):
    #with jit_scope():
    shp=tf.shape(kspace)
    scale=tf.sqrt(tf.dtypes.cast(shp[-2]*shp[-1],tf.float32 ))
    scale=tf.dtypes.cast(scale,tf.complex64)
    shifted=tf_ishift2d(kspace)
    xhat=tf.spectral.fft2d(shifted)/scale
    centered=tf_shift2d(xhat)
    return centered


def tf_ifft2c(kspace):
    #with jit_scope():
    shp=tf.shape(kspace)
    scale=tf.sqrt(tf.dtypes.cast(shp[-2]*shp[-1],tf.float32 ))
    scale=tf.dtypes.cast(scale,tf.complex64)
    shifted=tf_ishift2d(kspace)
    xhat=tf.spectral.ifft2d(shifted)*scale
    centered=tf_shift2d(xhat)
    return centered
#%% fftshifts on last two dimensions

def getIdx(x):
    xx=np.ceil(x/2).astype(np.int32)
    idx=np.concatenate( (range(xx,x),range(xx)),axis=0)
    return idx

def shift2d(img):
    x,y=img.shape[-2:]
    xid=getIdx(x)
    yid=getIdx(y)
    img=img[...,xid,:]
    img=img[...,yid]
    return img

def tf_getIdx(x):
    #with jit_scope():
    two=tf.constant(2)
    xx=tf.cast(tf.ceil(x/two),tf.int64)
    idx=tf.concat([tf.range(xx,x),tf.range(xx)],axis=0)
    return idx
def tf_shift2d(imgT):
    #with jit_scope():
    shp=tf.shape(imgT)
    x,y=shp[-2],shp[-1]
    xid=tf_getIdx(x)
    yid=tf_getIdx(y)
    imgT=tf.gather(imgT,xid,axis=-2)
    imgT=tf.gather(imgT,yid,axis=-1)
    return imgT


def tf_igetIdx(x):
    #with jit_scope():
    two=tf.constant(2)
    xx=tf.cast(tf.floor(x/two),tf.int64)
    idx=tf.concat([tf.range(xx,x),tf.range(xx)],axis=0)
    return idx

def tf_ishift2d(imgT):
    #with jit_scope():

    shp=tf.shape(imgT)
    x,y=shp[-2],shp[-1]
    xid=tf_igetIdx(x)
    yid=tf_igetIdx(y)
    imgT=tf.gather(imgT,xid,axis=-2)
    imgT=tf.gather(imgT,yid,axis=-1)
    return imgT


def tf_fftshift(x):
    shp=x.get_shape().as_list()[-2:]
    dim= [s//2 for s in shp]
    y=tf.manip.roll(x,dim,(-2,-1))
    return y


def tf_ifftshift(x):
    shp=x.get_shape().as_list()[-2:]
    dim= [(s+1)//2 for s in shp]
    y=tf.manip.roll(x,dim,(-2,-1))
    return y


def myfftshift(x):
    shp=x.shape[-2:]
    dim= [s//2 for s in shp]
    y=np.roll(x,dim,(-2,-1))
    return y


def myifftshift(x):
    shp=x.shape[-2:]
    dim= [(s+1)//2 for s in shp]
    y=np.roll(x,dim,(-2,-1))
    return y


#%%

def myPSNR(org,recon):
    sqrError=np.abs(org-recon)**2
    N=np.prod(org.shape[-2:])
    mse=np.sum(sqrError,axis=(-1,-2))/N
    maxval=np.max(org,axis=(-1,-2)) + 1e-15
    psnr=10*np.log10(maxval**2/(mse+1e-15 ))

    return psnr


#%%

def mySSIM(org,rec):
    """
    org and rec are 3D arrays in range [0,1]
    """
    shp=org.shape
    if np.ndim(org)>=3:
        nimg=np.prod(shp[0:-2])
    elif np.ndim(org)==2:
        nimg=1
    org=np.reshape(org,(nimg,shp[-2],shp[-1]))
    rec=np.reshape(rec,(nimg,shp[-2],shp[-1]))

    ssim=np.empty((nimg,),dtype=np.float32)
    for i in range(nimg):
        ssim[i]=structural_similarity(org[i],rec[i],data_range=org[i].max())
    return ssim


#%%

def r2c(inp):
    return inp[...,0] + 1j*inp[...,1]

def c2r(inp):
    return np.stack([np.real(inp),np.imag(inp)],axis=-1)

def tf_r2c(inp):
    return tf.complex(inp[...,0],inp[...,1])


def tf_c2r(inp):
    return tf.stack([tf.real(inp),tf.imag(inp)],axis=-1)


#%%
def getWeights(wtsDir,chkPointNum):
    """
    Input:
        wtsDir: Full path of directory containing modelTst.meta
    output:
        wt: numpy dictionary containing the weights. The keys names are full
        names of corersponding tensors in the model.
    """
    tf.reset_default_graph()
    if chkPointNum=='last':
        loadChkPoint=tf.train.latest_checkpoint(wtsDir)
    else:
        loadChkPoint=wtsDir+'/model'+chkPointNum
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as s1:
        saver = tf.train.import_meta_graph(wtsDir + '/modelTst.meta')
        saver.restore(s1, loadChkPoint)
        keys=[n.name+':0' for n in tf.get_default_graph().as_graph_def().node if "Variable" in n.op]
        var=tf.global_variables()

        wt={}
        for key in keys:
            va=[v for v in var if v.name==key][0]
            wt[key]=s1.run(va)

    tf.reset_default_graph()
    return wt

#%%
def assignWts(sess,wts):
    var=tf.global_variables()
    for v in var:
        if v.name in wts.keys():
            sess.run(v.assign(wts[v.name]))
    return sess

#%%
