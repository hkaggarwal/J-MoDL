"""
Created on Fri Aug 23 15:55:17 2019

@author: haggarwal
"""

import tensorflow as tf
import numpy as np
import misc as sf
from os.path import expanduser
home = expanduser("~")

def convLayer(x, szW,training,relu,i):
    with tf.name_scope('layers'):
        with tf.variable_scope('lay'+str(i)):
            init=tf.contrib.layers.xavier_initializer()
            W=tf.get_variable('W',shape=szW,initializer=init,trainable=True)
            y = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
            #y=tf.layers.batch_normalization(y,training=training,trainable=True,fused=True,name='BN')
            if relu:
                y=tf.nn.relu(y)
    return y


def bigUnet(inp,C,training):
    with tf.name_scope('Unet'):
        x=convLayer(inp,(3,3,C,64),training,True,1)
        x1=convLayer(x,(3,3,64,64),training,True,2)
        p1=tf.nn.avg_pool(x1,[1,2,2,1],[1,2,2,1],'SAME')

        x=convLayer(p1,(3,3,64,128),training,True,3)
        x2=convLayer(x,(3,3,128,128),training,True,4)
        p2=tf.nn.avg_pool(x2,[1,2,2,1],[1,2,2,1],'SAME')

        x=convLayer(p2,(3,3,128,256),training,True,5)
        x3=convLayer(x,(3,3,256,256),training,True,6)
        p3=tf.nn.avg_pool(x3,[1,2,2,1],[1,2,2,1],'SAME')

        x=convLayer(p3,(3,3,256,512),training,True,7)
        x4=convLayer(x,(3,3,512,512),training,True,8)
        p4=tf.nn.avg_pool(x4,[1,2,2,1],[1,2,2,1],'SAME')

        x=convLayer(p4,(3,3,512,512),training,True,9)

        shp=tf.shape(x4)
        x=tf.image.resize_bilinear(x,[shp[-3],shp[-2]])
        x=tf.concat([x4,x],axis=-1)
        x=convLayer(x,(3,3,1024,256),training,True,10)
        x=convLayer(x,(3,3,256,256),training,True,11)

        shp=tf.shape(x3)
        x=tf.image.resize_bilinear(x,[shp[-3],shp[-2]])
        x=tf.concat([x3,x],axis=-1)
        x=convLayer(x,(3,3,512,128),training,True,12)
        x=convLayer(x,(3,3,128,128),training,True,13)

        shp=tf.shape(x2)
        x=tf.image.resize_bilinear(x,[shp[-3],shp[-2]])
        x=tf.concat([x2,x],axis=-1)
        x=convLayer(x,(3,3,256,64),training,True,14)
        x=convLayer(x,(3,3,64,64),training,True,15)

        shp=tf.shape(x1)
        x=tf.image.resize_bilinear(x,[shp[-3],shp[-2]] )
        x=tf.concat([x1,x],axis=-1)
        x=convLayer(x,(3,3,128,64),training,True,16)
        x=convLayer(x,(3,3,64,64),training,True,17)

        x=convLayer(x,(1,1,64,C),training,False,18)


    return x

def smallModel(inp,c,training):
    fs=3 #filter size
    with tf.name_scope('7layer'):
        x=convLayer(inp,(fs,fs,c ,64),training,True,1)
        x=convLayer(x,(fs,fs,64,64),training,True,2)
        x=convLayer(x,(fs,fs,64,128),training,True,3)
        x=convLayer(x,(fs,fs,128,128),training,True,4)
        x=convLayer(x,(fs,fs,128,64),training,True,5)
        x=convLayer(x,(fs,fs,64,64),training,True,6)
        x=convLayer(x,(1,1,64,c),'linear',False,7)

    return x

def myCG(A,rhs,cgIter,cgTol):

    cond=lambda i,rTr,*_: tf.logical_and( tf.less(i,cgIter), tf.abs(rTr)>cgTol)
    fn=lambda a,b:tf.reduce_sum(tf.conj(a)*b)
    def body(i,rTr,x,r,p):
        with tf.name_scope('cgBody'):
            Ap=A(p)
            alpha = rTr / fn(p,Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = fn(r,r)
            beta = rTrNew / rTr
            p = r + beta * p
        return i+1,rTrNew,x,r,p

    x=tf.zeros_like(rhs)
    i,r,p=0,rhs,rhs
    rTr =  fn(r,r)
    loopVar=i,rTr,x,r,p
    out=tf.while_loop(cond,body,loopVar,name='CGwhile',parallel_iterations=1,
                      maximum_iterations=cgIter)[2]
    return out

def myAtA(img,csm,atah,aatv):
    cimg=csm*img
    tmp= atah@cimg@aatv
    coilComb= tf.reduce_sum(tmp*tf.conj(csm),axis=-3)
    return coilComb

def funDc(z,atah,aatv,atbT,csmT,lamT,cgIter,cgTol):
    #with jit_scope():
    z=sf.tf_r2c(z)
    rhs=atbT+lamT*z
    def fn(inp):
        c,r=inp
        AtA=lambda x: myAtA(x,c,atah,aatv)
        B = lambda x: AtA(x)+lamT*x
        y=myCG(B,r,cgIter,cgTol)
        return y
    inp=(csmT,rhs)
    rec=tf.map_fn(fn,inp,dtype=tf.complex64,name='mapFn' )
    rec=sf.tf_c2r(rec)
    return rec

def funDw(xin,training):
    with tf.name_scope('dwBlk'):
        inp=xin
        #inp,mean,std=sf.tf_cpxNormalizeMeanStd(xin)
        with tf.variable_scope('laye',reuse=tf.AUTO_REUSE):
            #nw=bigUnet(inp,2,training)
            nw=smallModel(inp,2,training)
            dw=nw+inp
        #dw=dw*std+mean
    return dw

def tf_getAAt(m,N):
    n=tf.range(N)
    jTwoPi=tf.constant(1j*2*np.pi,dtype=tf.complex64)
    scale= tf.constant(1./np.sqrt(N),dtype=tf.complex64)

    m=tf.cast(m,tf.complex64)
    n=tf.cast(n,tf.complex64)

    A=tf.exp(-jTwoPi*(m-1/2)*(n-N/2))*scale
    At=tf.transpose(A,conjugate=True)
    return A,At

def consFn(x):
    y=tf.clip_by_value(x,0,1)
    return y



#%%

def makeModel(orgT,csmT,initx,inity,M,N,sigma,lam,K,training):
    cgIter=tf.constant(10,name='cgIter')
    cgTol=tf.constant(1e-10,name='cgTol',dtype=tf.float32)
    lamT= tf.constant(lam, dtype=tf.float32,name='lam')
    lamT=tf.complex(lamT,0.)

    kx=tf.get_variable('kx',dtype=tf.float32, trainable=True,
                   constraint=consFn, initializer=initx)
    ky=tf.get_variable('ky',dtype=tf.float32, trainable=True,
                       constraint=consFn, initializer=inity)
    csmconj=tf.conj(csmT)
    Ah,Aht=tf_getAAt(kx,M)
    Av,Avt=tf_getAAt(ky,N)
    Av=tf.transpose(Av)
    Avt=tf.transpose(Avt)
    atah=Aht@Ah
    aatv=Av@Avt
    A= lambda x: Ah@(csmT*x[:,tf.newaxis])@Av
    At=lambda x: tf.reduce_sum(csmconj*(Aht@x@Avt),axis=-3)

    bT=A(orgT)
    shp=tf.shape(bT)
    sigmaT=tf.cast(sigma,tf.complex64)
    noiseT=tf.complex(tf.random_normal(shp),tf.random_normal(shp))*sigmaT
    bT=bT+noiseT
    atbT=At(bT)


    with tf.name_scope('model'):

        Dw=lambda x: funDw(x,training)
        Dc=lambda x: funDc(x,atah,aatv,atbT,csmT,lamT,cgIter,cgTol)

        z=tf.zeros_like(atbT)
        z=sf.tf_c2r(z)
        x=Dc(z)
        for i in range(K):
            z=Dw(x)
            x=Dc(z)

    return atbT,x

