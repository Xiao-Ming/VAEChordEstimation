#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:16:45 2019

@author: wuyiming
"""


from chainer import Chain,kl_divergence,reporter,serializers,Sequential
import chainer.functions as F
import chainer.links as L
import chainer.distributions as D
import util as U

import const as C

class Classifier_RNN(Chain):
    def __init__(self):
        super(Classifier_RNN,self).__init__()
        with self.init_scope():
            self.rnn = L.NStepBiLSTM(3,C.N_DIMS_FEAT,C.N_DIMS_CLASSIFIER_LATENT,0.0)
            self.li_rnn = L.Linear(C.N_DIMS_CLASSIFIER_LATENT*2,C.N_DIMS_CLASSIFIER_LATENT,nobias=True)
            self.norm = L.LayerNormalization(C.N_DIMS_CLASSIFIER_LATENT)
            self.li = L.Linear(C.N_DIMS_CLASSIFIER_LATENT,C.N_VOCABULARY_TRIADS)
            
    def __call__(self,xs,*args):
        _,_,hs = self.rnn(None,None,xs)
        hs = [self.norm(self.li_rnn(h)) for h in hs]
        ys = [self.li(h) for h in hs]
        return ys

"""
class Classifier_RNN_Normal(Chain):
    def __init__(self):
        super(Classifier_RNN_Normal,self).__init__()
        with self.init_scope():
            self.rnn = L.NStepBiLSTM(3,C.N_DIMS_FEAT,C.N_DIMS_CLASSIFIER_LATENT,0.0)
            self.li_rnn = L.Linear(C.N_DIMS_CLASSIFIER_LATENT*2,C.N_DIMS_CLASSIFIER_LATENT,nobias=True)
            self.norm = L.LayerNormalization(C.N_DIMS_CLASSIFIER_LATENT)
            self.li = L.Linear(C.N_DIMS_CLASSIFIER_LATENT,C.N_VOCABULARY_TRIADS*2)
            
    def __call__(self,xs,*args):
        _,_,hs = self.rnn(None,None,xs)
        hs = [self.norm(self.li_rnn(h)) for h in hs]
        ys = [F.split_axis(self.li(h),2,-1) for h in hs]
        return ys



class Classifier_Attend(Chain):
    def __init(self):
        super(Classifier_Attend,self).__init__()
        with self.init_scope():
            self.li_in = L.Linear(C.N_DIMS_FEAT,C.N_DIMS_RNN_LATENT,noias=True)
            self.attend = Sequential(SelfAttention(),
                                     SelfAttention(),
                                     SelfAttention(),
                                     SelfAttention())
            self.li_out = L.Linear(C.N_DIMS_RNN_LATENT,C.N_VOCABULARY_TRIADS)
    def __call__(self,xs,*args):
        x = F.stack(xs)
        h = self.li_in(x,n_batch_axes=2)
        h = self.attend(h)
        ys = F.separate(self.li_out(h,n_batch_axes=2))
        return ys


class Classifier_RNN_Normalized(Chain):
    def __init__(self):
        super(Classifier_RNN_Normalized,self).__init__()
        with self.init_scope():
            self.rnn1 = L.NStepBiLSTM(1,C.N_DIMS_FEAT,C.N_DIMS_CLASSIFIER_LATENT,0.0)
            self.norm1 = L.LayerNormalization(C.N_DIMS_CLASSIFIER_LATENT*2)
            self.rnn2 = L.NStepBiLSTM(1,C.N_DIMS_CLASSIFIER_LATENT*2,C.N_DIMS_CLASSIFIER_LATENT,0.0)
            self.norm2 = L.LayerNormalization(C.N_DIMS_CLASSIFIER_LATENT*2)
            self.rnn3 = L.NStepBiLSTM(1,C.N_DIMS_CLASSIFIER_LATENT*2,C.N_DIMS_CLASSIFIER_LATENT,0.0)
            self.norm3 = L.LayerNormalization(C.N_DIMS_CLASSIFIER_LATENT*2)
            self.li = L.Linear(C.N_DIMS_CLASSIFIER_LATENT*2,C.N_VOCABULARY_TRIADS)
    def __call__(self,xs,*args):
        _,_,hs = self.rnn1(None,None,xs)
        hs = [self.norm1(h) for h in hs]
        _,_,hs = self.rnn2(None,None,hs)
        hs = [self.norm2(h) for h in hs]
        _,_,hs = self.rnn3(None,None,hs)
        ys = [self.li(self.norm3(h)) for h in hs]
        return ys
        
class ClassifierCNNFramewise(Chain):
    def __init__(self):
        super(ClassifierCNNFramewise,self).__init__()
        with self.init_scope():
            self.conv1 = L.LocalConvolution2D(2,32,in_size=None,ksize=(3,3))
            self.norm1 = L.BatchRenormalization(32)
            self.conv2 = L.LocalConvolution2D(32,32,in_size=None,ksize=(5,5))
            self.norm2 = L.BatchRenormalization(32)
            #(Max pooling 1x2)
            
            self.conv3 = L.Convolution2D(32,256,ksize=(3,3))
            self.norm3 = L.BatchRenormalization(256)
            self.conv4 = L.Convolution2D(256,256,ksize=(3,4))
            self.norm4 = L.BatchRenormalization(256)
            #(Max pooling 1x2)
            self.conv5 = L.Convolution2D(256,C.N_VOCABULARY_TRIADS,ksize=(1,1))
            #(Avg.pooling)
    def __call__(self,xs):
        h = self.norm1(F.leaky_relu(self.conv1(F.expand_dims(xs,axis=1))))
        h = self.norm2(F.leaky_relu(self.conv2(h)))
        h = F.max_pooling_2d(h,(1,2))
        h = self.norm3(F.leaky_relu(self.conv3(h)))
        h = self.norm4(F.leaky_relu(self.conv4(h)))
        h = F.max_pooling_2d(h,(1,2))
        h = F.average_pooling_2d(self.conv5(h),(5,17))
        assert h.shape[1:]== (C.N_VOCABULARY_TRIADS,1,1), "Wrong output shape: h.shape = %s" % str(h.shape)
        return F.squeeze(h)
            
class ConvNet(Chain):
    def __init__(self):
        super(ConvNet,self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=1,out_channels=32,ksize=3,stride=1,pad=1)
            self.norm1 = L.BatchNormalization(32)
            self.conv2 = L.Convolution2D(in_channels=32,out_channels=32,ksize=3,stride=1,pad=1)
            self.norm2 = L.BatchNormalization(32)
            self.conv3 = L.Convolution2D(in_channels=32,out_channels=32,ksize=3,stride=1,pad=1)
            self.norm3 = L.BatchNormalization(32)
            self.conv4 = L.Convolution2D(in_channels=32,out_channels=32,ksize=3,stride=1,pad=1)
            self.norm4 = L.BatchNormalization(32)
            self.conv5 = L.Convolution2D(in_channels=32,out_channels=64,ksize=3,stride=1,pad=0)
            self.norm5 = L.BatchNormalization(64)
            self.conv6 = L.Convolution2D(in_channels=64,out_channels=64,ksize=3,stride=1,pad=0)
            self.norm6 = L.BatchNormalization(64)
            self.conv7 = L.Convolution2D(in_channels=64,out_channels=128,ksize=(9,12),stride=1,pad=0)
            self.norm7 = L.BatchNormalization(128)
            self.conv8 = L.Convolution2D(in_channels=128,out_channels=C.N_VOCABULARY_TRIADS,ksize=1,stride=1,pad=0)
            self.norm8 = L.BatchNormalization(C.N_VOCABULARY_TRIADS)
            
    def __call__(self,x):
        h1 = self.norm1(F.relu(self.conv1(F.expand_dims(x,axis=1))))
        h2 = self.norm2(F.relu(self.conv2(h1)))
        h3 = self.norm3(F.relu(self.conv3(h2)))
        h4 = self.norm4(F.relu(self.conv4(h3)))
        h5 = F.dropout(F.max_pooling_2d(h4,ksize=(1,2)))
        h6 = self.norm5(F.relu(self.conv5(h5)))
        h7 = self.norm6(F.relu(self.conv6(h6)))
        h8 = F.dropout(F.max_pooling_2d(h7,ksize=(1,2)))
        h9 = F.dropout(self.norm7(F.relu(self.conv7(h8))))
        h10 = self.norm8(self.conv8(h9))
        out = F.squeeze(F.average_pooling_2d(h10,ksize=(3,7)))
        assert out.shape[1:]== (C.N_VOCABULARY_TRIADS,), "Wrong output shape: h.shape = %s" % str(out.shape)
        
        return out
    def getFeat(self,x):
        h1 = self.norm1(F.relu(self.conv1(x)))
        h2 = self.norm2(F.relu(self.conv2(h1)))
        h3 = self.norm3(F.relu(self.conv3(h2)))
        h4 = self.norm4(F.relu(self.conv4(h3)))
        h5 = F.dropout(F.max_pooling_2d(h4,ksize=(1,2)))
        h6 = self.norm5(F.relu(self.conv5(h5)))
        h7 = self.norm6(F.relu(self.conv6(h6)))
        h8 = F.dropout(F.max_pooling_2d(h7,ksize=(1,2)))
        h9 = F.dropout(self.norm7(F.relu(self.conv7(h8))))
        out = F.average_pooling_2d(h9,ksize=(3,23))[:,:,0,0]
        return out

    def save(self,fname="convnet.model"):
        serializers.save_npz(fname,self)
    def load(self,fname="convnet.model"):
        serializers.load_npz(fname,self)

class Classifier_Spec(Chain):
    def __init__(self):
        super(Classifier_Spec,self).__init__()
        with self.init_scope():
            self.spec_conv1 = L.Convolution2D(1,32,(7,3),1,(3,1))
            self.spec_norm1 = L.BatchNormalization(32)
            self.spec_conv2 = L.DepthwiseConvolution2D(32,1,(3,3),1,(1,1))
            self.spec_norm2 = L.BatchNormalization(32)
            #Max pooling(3,3)
            self.spec_conv3 = L.DepthwiseConvolution2D(32,2,(3,3),1,(1,1))
            self.spec_norm3 = L.BatchNormalization(64)
            self.spec_conv4 = L.DepthwiseConvolution2D(64,1,(3,3),1,(1,1))
            self.spec_norm4 = L.BatchNormalization(64)
            #Max pooling(3,4)
            self.spec_conv5 = L.Convolution2D(64,C.N_VOCABULARY_TRIADS,(7,C.N_CQT_BINS//12),1,(3,0))
            #self.spec_norm5 = L.BatchNormalization(64)
            #self.spec_rnn = L.NStepBiLSTM(1,64,C.N_DIMS_RNN_LATENT,0.0)
            
            #self.feat_rnn = L.NStepBiLSTM(4,C.N_DIMS_FEAT,C.N_DIMS_RNN_LATENT,0.0)
            
            #self.dense1 = L.Linear(C.N_DIMS_RNN_LATENT*2,C.N_VOCABULARY_TRIADS)
    
    def __call__(self,xs):
        xs = F.expand_dims(xs,1)
        h_specs = F.leaky_relu(self.spec_norm1(self.spec_conv1(xs)))
        h_specs = F.leaky_relu(self.spec_norm2(self.spec_conv2(h_specs)))
        h_specs = F.max_pooling_2d(h_specs, ksize=3, stride=(1,3), pad=(1,0))
        h_specs = F.leaky_relu(self.spec_norm3(self.spec_conv3(h_specs)))
        h_specs = F.leaky_relu(self.spec_norm4(self.spec_conv4(h_specs)))
        h_specs = F.max_pooling_2d(h_specs, ksize=(3,4), stride=(1,4), pad=(1,0))
        h_specs = F.transpose(F.squeeze(self.spec_conv5(h_specs),axis=3),axes=(0,2,1))
        #_, _, h_specs = self.spec_rnn(None, None, F.separate(h_specs,axis=0))
        
        #_, _, h_feats = self.feat_rnn(None, None, feats)
        
        #h_specs = F.stack(h_specs,axis=0)
        #h_out = self.dense1(h_specs,n_batch_axes=2)
        return h_specs
    def estimate_labels(self,xs,*args):
        ys = self(xs)
        labs = F.argmax(ys,axis=-1)
        return labs


class ResBlock(Chain):
    def __init__(self,n_channels,ksize,stride,pad):
        super(ResBlock,self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(n_channels,n_channels,ksize,stride,pad)
            self.conv2 = L.Convolution2D(n_channels,n_channels,ksize,stride,pad)
            self.norm2 = L.BatchNormalization(n_channels)
            
    
    def __call__(self,x):
        h = F.relu(self.conv1(x))
        h = self.norm2(F.relu(self.conv2(h)))
        return x+h

class CRNNClassifier(Chain):
    def __init__(self,n_resblocks=2):
        super(CRNNClassifier,self).__init__()
        with self.init_scope():
            self.spec_conv1 = L.Convolution2D(5,64,(15,3),1,(7,1))
            self.spec_norm1 = L.BatchNormalization(64)
            self.spec_conv2 = L.Convolution2D(64,128,(3,3),1,(1,1))
            self.spec_norm2 = L.BatchNormalization(128)
            #Max pooling (3,3)
            self.res = Sequential(*[ResBlock(128,5,1,2) for i in range(n_resblocks)]) 
            #Max pooling (3,4)
            self.spec_conv3 = L.Convolution2D(128,128,(15,C.N_CQT_BINS//12),1,(7,0))
            self.spec_norm3 = L.BatchNormalization(128)
            
            self.spec_rnn = L.NStepBiLSTM(2,128,C.N_DIMS_RNN_LATENT,0.5)
            self.dense1 = L.Linear(C.N_DIMS_RNN_LATENT*2,C.N_VOCABULARY_TRIADS)
    
    def __call__(self,specs):
        specs = F.stack(specs, axis=0)
        h_specs = self.spec_norm1(F.leaky_relu(self.spec_conv1(specs)))
        h_specs = self.spec_norm2(F.leaky_relu(self.spec_conv2(h_specs)))
        h_specs = F.max_pooling_2d(h_specs, ksize=3, stride=(1,3), pad=(1,0))
        h_specs = self.res(h_specs)
        
        h_specs = F.max_pooling_2d(h_specs, ksize=(3,4), stride=(1,4), pad=(1,0))
        h_specs = F.transpose(F.squeeze(self.spec_norm3(F.relu(self.spec_conv3(h_specs))),axis=3),axes=(0,2,1))
        #assert h_specs.shape == (C.N_BATCH,C.SEQLEN_TRAINING,128), "Invalid shape of h_specs: %s" % str(h_specs.shape)
        _, _, h_specs = self.spec_rnn(None, None, F.separate(h_specs,axis=0))
        
        h_out = [self.dense1(h) for h in h_specs]
        #h_out = F.separate(h_out,axis=0)
        
        return h_out  
"""
