#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:11:27 2019

@author: wuyiming
"""

from chainer import Chain,Link,Distribution,Variable,kl_divergence,reporter,serializers,as_variable,variable,Sequential,FunctionNode
import chainer.initializers as I
import chainer.functions as F
import chainer.links as L
import chainer.distributions as D
import numpy as np
from chord import template_matrix,template_matrix_zero_center,chroma_shifter,label_shifter,bass_shifter

import const as C
import util as U





from itertools import chain

class ChromaVAE(Chain):
    def __init__(self,k=1,pseudo_dset=None):
        super(ChromaVAE,self).__init__()
        with self.init_scope():
            self.encoder = Encoder_normal()
            #self.encoder = Encoder_conv_normal()
            #self.encoder = Encoder_echo()
            #self.decoder = Decoder_CNN_sigm()
            #self.decoder = Decoder_Unetconv_sigm()
            self.decoder = Decoder_normal()
            #self.decoder = Decoder_CNN_sigm()
            self.prior = PriorNormal(C.N_DIMS_VAE_LATENT)
            #self.discriminator = LatentDiscriminator()

        if pseudo_dset is not None:
            self.vamp_prior = True
            self.pseudo_dset = pseudo_dset
        else:
            self.vamp_prior = False
        
        self.k = k
        self.beta = 1.0
        self.label_shifter = label_shifter
        self.chroma_shifter = chroma_shifter
        self.register_persistent("label_shifter")
        self.register_persistent("chroma_shifter")
    def _encode_onehot(self,y,n_category=C.N_VOCABULARY_TRIADS,noise=False,xp=np):
        code = xp.zeros((len(y),n_category),dtype=xp.float32)
        code[xp.arange(code.shape[0]),y] = 1
        return code
    def _shift_label(self,y,shift):
        if shift==0:
            return y
        rollidx = np.roll(np.arange(12),shift)
        ys_shift = F.concat((y[:,rollidx],
                             y[:,rollidx+12],
                             y[:,rollidx+24],
                             y[:,rollidx+36],
                             y[:,rollidx+48],
                             y[:,rollidx+60],
                             y[:,72:]),axis=1)
        return ys_shift

    def _shift_feature(self,x,shift):
        if shift==0:
            return x
        segments = x.shape[1] // 12
        seg_list = []
        for i in range(segments):
            seg_list.append(self.xp.roll(x[:,i*12:(i+1)*12],shift,1))
        feat_shifted = self.xp.concatenate(seg_list,axis=1)
        return feat_shifted

    def _shift_label_quality(self,y,target_lab):
        y_argmax = F.argmax(y,1).data
        shiftmat = self.label_shifter[y_argmax,target_lab,:,:]
        ys_shift = y[:,None,:] @ shiftmat
        return F.squeeze(ys_shift,1)

    def _shift_feature_quality(self,x,y,target_lab):
        y_argmax = F.argmax(y,1).data
        shiftmat = self.chroma_shifter[y_argmax,target_lab,:C.N_DIMS_FEAT,:C.N_DIMS_FEAT]
        xs_shift = x[:,None,:] @ shiftmat
        return F.squeeze(xs_shift,1)

    def __call__(self,xs,ys,aligns=None,anneal_step=None,no_shift=False):
        batch_size = len(xs)
        if C.VAE_RAND_SHIFT:
            randshift = np.random.randint(0,12)
            _xs = [self._shift_feature(x,randshift) for x in xs]
            _ys = [self._shift_label(y,randshift) for y in ys]
        elif C.VAE_RAND_SHIFT_QUALITY:
            randshift = np.random.randint(12)
            targ = [F.argmax(y,1).data for y in ys]
            for b in range(len(targ)):
                for l in range(24):
                    targ[b][targ[b]==l] = (12*np.random.randint(6)) + (l%12)
                idx = targ[b]!=C.LABEL_N
                targ[b][idx] = targ[b][idx]//12*12 +(targ[b][idx] + randshift) % 12
            _xs = [self._shift_feature_quality(x,y,t) for x,y,t in zip(xs,ys,targ)]
            _ys = [self._shift_label_quality(y,t) for y,t in zip(ys,targ)]
        else:
            randshift = 0
            _ys = ys
            _xs = xs
        dist_z = self.encoder(xs,ys)
        #zs,mi = self.encoder(xs,ys)
        #prior_z = [D.Normal(self.xp.ones(z.loc.shape,dtype=self.xp.float32),self.xp.zeros(z.scale.shape,dtype=self.xp.float32)) for z in dist_z]
        if self.vamp_prior:
            pseudo_xs = []
            pseudo_ys = []
            for i in range(len(self.pseudo_dset)):
                x,y,align = self.pseudo_dset[i]
                pseudo_xs.append(self.xp.asarray(x))
                pseudo_ys.append(self.xp.asarray(self._encode_onehot(y)[align]))
            pseudo_zs = self.encoder(pseudo_xs,pseudo_ys)
            mean_loc = sum([F.mean(d.loc,axis=0) for d in pseudo_zs]) / len(self.pseudo_dset)
            mean_scale = sum([F.mean(d.log_scale,axis=0)*2 for d in pseudo_zs]) / len(self.pseudo_dset) * 2
            #prior_z = [D.Independent(D.MultivariateNormal(loc=mean_loc,scale_tril=mean_scale)) for i in range(len(self.pseudo_dset))]
            prior_z = [D.Normal(loc=mean_loc,log_scale=mean_scale) for i in range(len(self.pseudo_dset))]
        else:
            #pass
            prior_z = [self.prior(z.distribution.loc.shape) for z in dist_z]
        #prior_z = self.prior(dist_z.shape)
        #prior_z = self.prior(ys,aligns)
        zs = [d.sample_n(self.k)[0,:] for d in dist_z]
        #xs_bass_dec,xs_mid_dec,xs_top_dec = self.decoder(zs,_ys,aligns)
        xs_dec = self.decoder(zs,_ys)

        #kl_penalty = F.mean(F.sum(kl_divergence(dist_z,prior_z),axis=-1))
        #reconstr = F.mean(F.sum(xs_dec.log_prob(F.stack(xs)),axis=-1))
        kl_term = sum([F.mean(kl_divergence(d1,d2)) for d1,d2 in zip(dist_z,prior_z)]) / batch_size
        #kl_penalty = mi
        """
        reconstr = sum([F.mean(x_dec.log_prob(x[:,:13])) for x_dec,x in zip(xs_bass_dec,_xs)]) / batch_size + \
                   sum([F.mean(F.sum(x_dec.log_prob(x[:,13:25]),axis=-1)) for x_dec,x in zip(xs_mid_dec,_xs)]) / batch_size + \
                   sum([F.mean(x_dec.log_prob(x[:,25:])) for x_dec,x in zip(xs_top_dec,_xs)]) / batch_size
        """
        log_p_x_s_z = sum([F.mean(x_dec.log_prob(x)) for x_dec,x in zip(xs_dec,_xs)]) / batch_size
        #reconstr = 0.0
        #reconstr = sum([F.mean(F.sum(-x*F.log(x_dec) - (1-x)*F.log(1-x_dec),axis=-1)) for x_dec,x in zip(xs_dec,xs)]) / batch_size
        
        #loss = -(reconstr - (C.VAE_BETA*kl_penalty))
        if anneal_step is not None:
            beta = C.VAE_BETA_ANNEAL_TO * (anneal_step**2) / 20000
            beta = min([beta,C.VAE_BETA_ANNEAL_TO])
        else:
            beta = C.VAE_BETA
        loss = -log_p_x_s_z + (beta * kl_term)
        reporter.report({"loss":loss}, self)
        reporter.report({"reconstr":log_p_x_s_z}, self)
        reporter.report({"kl":kl_term}, self)
        reporter.report({"beta":beta}, self)
        return loss

    def loss_discrim(self,xs,ys):
        dist_zs = self.encoder(xs,ys)
        zs = [F.mean(d.sample_n(self.k),0) for d in dist_zs]
        discrim_loss = self.discriminator([z.data for z in zs],ys)
        return discrim_loss

    def reconstr_dist(self,xs,ys,aligns=None,shift=0):
        dist_z = self.encoder(xs,ys)
        zs = [d.sample_n(self.k)[0] for d in dist_z]
        #zs,_ = self.encoder(xs,ys)
        #xs_dec_bass,xs_dec_mid,xs_dec_top = self.decoder(zs,ys)
        #xs_dec = [F.concat([b.p,m.p,t.p],axis=-1).data for b,m,t in zip(xs_dec_bass,xs_dec_mid,xs_dec_top)]
        xs_dec = self.decoder(zs,ys)
        return [d.distribution.p.data for d in xs_dec]

    def reconstr_dist_randzs(self,xs,ys,aligns=None):
        np.random.seed(100)
        #dist_z = self.encoder(_xs,ys,aligns)
        #zs = list(chain.from_iterable([F.separate(d.sample(self.k)) for d in dist_z]))
        zs = [self.xp.random.normal(0,1,size=(x.shape[0],C.N_DIMS_VAE_LATENT)).astype(self.xp.float32) for x in xs]
        #zs = [self.xp.zeros((x.shape[0],C.N_DIMS_VAE_LATENT)).astype(self.xp.float32) for x in xs]
        xs_dec = self.decoder(zs,ys,aligns)
        return xs_dec

    def getzs(self,xs,ys,aligns=None):
        dist_z = self.encoder(xs,[self._encode_onehot(y,xp=self.xp) for y in ys],aligns)
        zs = list(chain.from_iterable([F.separate(d.sample(self.k)) for d in dist_z]))
        return zs
    
    
    def generate_condition(self,ys,aligns):
        prior_z = self.prior((aligns.size,C.N_DIMS_VAE_LATENT))
        z = prior_z.sample_n(1)[0]
        x_dec = self.decoder([z],[ys],[aligns])[0]
        #return x_dec.p
        #return x_dec.sample(1)[0,:,:]
        return x_dec.a/(x_dec.a+x_dec.b)
    
    def generate_encode_condition(self,xs,ys,aligns):
        if aligns is not None:
            aligns = [aligns]
        dist_z = self.encoder([xs],[ys],aligns)[0]
        z = dist_z.sample(1)[0,:,:]
        x_dec = self.decoder([z],[ys],aligns)[0]
        #return x_dec.p
        #return x_dec.sample(1)[0,:,:]
        return x_dec.a/(x_dec.a+x_dec.b)    
    def save(self,path):
        serializers.save_npz(path,self)
    def load(self,path):
        serializers.load_npz(path,self)



        
class Encoder_normal(Chain):
    def __init__(self):
        super(Encoder_normal,self).__init__()
        with self.init_scope():
            #self.rnn = L.NStepBiLSTM(2,C.N_DIMS_FEAT+C.N_VOCABULARY_TRIADS,C.N_DIMS_RNN_LATENT,0.0)
            self.rnn = L.NStepBiLSTM(2,C.N_DIMS_FEAT,C.N_DIMS_RNN_LATENT,0.0)
            self.li_rnn = L.Linear(C.N_DIMS_RNN_LATENT*2,C.N_DIMS_RNN_LATENT,nobias=True)
            self.norm = L.LayerNormalization(C.N_DIMS_RNN_LATENT)
            self.mu = L.Linear(C.N_DIMS_RNN_LATENT,C.N_DIMS_VAE_LATENT)
            self.sigma = L.Linear(C.N_DIMS_RNN_LATENT,C.N_DIMS_VAE_LATENT)
        self.chroma_shifter = chroma_shifter
        self.register_persistent("chroma_shifter")

    def forward(self,xs,ys=None,aligns=None):
        if C.VAE_SHIFT_REGULAR:
            ys_argmax = [F.argmax(y,1).data for y in ys]
            _xs = [F.squeeze(x[:,None,:] @ self.chroma_shifter[y,0,:,:],1) for x,y in zip(xs,ys_argmax)]
        else:
            _xs = xs
        #xs_ys = [F.concat((x,y),axis=1) for x,y in zip(_xs,ys)]
        #xs_ys = ys_aligned
        _,_,hs = self.rnn(None,None,_xs)
        hs = [self.norm(self.li_rnn(h)) for h in hs]
        mu = [self.mu(h) for h in hs]
        sigma = [self.sigma(h) for h in hs]
        #return [D.Independent(D.MultivariateNormal(loc=m,scale_tril=s),1) for m,s in zip(mu,sigma)]
        return [D.Independent(D.Normal(loc=m,log_scale=s)) for m,s in zip(mu,sigma)]



  
class Decoder_normal(Chain):
    def __init__(self):
        super(Decoder_normal,self).__init__()
        with self.init_scope():
            self.rnn = L.NStepBiLSTM(3,C.N_DIMS_VAE_LATENT+C.N_VOCABULARY_TRIADS,C.N_DIMS_RNN_LATENT,0.0)
            #self.rnn = L.NStepBiLSTM(3,C.N_VOCABULARY_TRIADS,C.N_DIMS_RNN_LATENT,0.0)
            self.li_rnn = L.Linear(C.N_DIMS_RNN_LATENT*2,C.N_DIMS_RNN_LATENT,nobias=True)
            self.norm = L.LayerNormalization(C.N_DIMS_RNN_LATENT)
            self.li_mu = L.Linear(C.N_DIMS_RNN_LATENT,C.N_DIMS_FEAT)
            #self.li_out = L.Linear(C.N_DIMS_RNN_LATENT,13+12+13)

    def forward(self,zs,ys,aligns=None):
        if aligns is not None:
            ys_aligned = [y[a,:] for y,a in zip(ys,aligns)]
        else:
            ys_aligned = ys
        zs_ys = [F.concat((z,y),axis=1) for z,y in zip(zs,ys_aligned)]
        _,_,hs = self.rnn(None,None,zs_ys)
        hs = [self.norm(self.li_rnn(h)) for h in hs]
        hs = [self.li_mu(h) for h in hs]
        #hs_out = [F.split_axis(self.li_out(h),[13,25],axis=-1) for h in hs]
        #dist_mid = [D.Bernoulli(logit=h[1]) for h in hs_out]
        #dist_bass = [D.OneHotCategorical(p=F.softmax(h[0],-1)) for h in hs_out]
        #dist_top = [D.OneHotCategorical(p=F.softmax(h[2],-1)) for h in hs_out]
        
        dist = [D.Independent(D.Bernoulli(logit=h)) for h in hs]

        return dist



class PriorNormal(Link):
    def __init__(self, shape):
        super(PriorNormal, self).__init__()
        self.loc = np.zeros(shape, np.float32)
        self.scale = np.ones(shape, np.float32)
        self.register_persistent('loc')
        self.register_persistent('scale')

    def forward(self,shape):
        return D.Independent(D.Normal(F.broadcast_to(self.loc,shape), scale=F.broadcast_to(self.scale,shape)))
    



#DECODER_CLASS = {"lstm":Decoder_normal,
#                 "cnn":Decoder_CNN_sigm,
#                 "unet":Decoder_Unetconv_sigm}

