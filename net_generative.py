#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:27:50 2019

@author: wuyiming
"""

from chainer import Chain,Function,Parameter,Link,kl_divergence,reporter,serializers,Variable,initializers,FunctionNode,using_config
import chainer.functions as F
import chainer.links as L
import chainer.distributions as D
import net_chordrec
import vae
import util as U
import numpy as np
import const as C
from librosa.sequence import viterbi as vtb
from librosa.sequence import transition_loop



"""

def JSD(d1,d2):
    def kl(_d1,_d2):
        return F.sum(_d1*F.log(_d1/_d2),axis=1)
    M = (d1+d2) / 2.
    jsd = kl(d1,M)/2. + kl(d2,M)/2.
    return F.mean(jsd)


class PriorCategorical(Link):
    def __init__(self, dist=None):
        super(PriorCategorical, self).__init__()
        if dist is None:
            self.p = np.ones(C.N_VOCABULARY_TRIADS, np.float32) / C.N_VOCABULARY_TRIADS
        else:
            self.p = dist
        self.register_persistent('p')

    def forward(self,shape):
        return D.Categorical(F.broadcast_to(self.p,shape))

class PriorDirichlet(Link):
    def __init__(self,classes):
        super(PriorDirichlet,self).__init__()
        self.p = np.ones(classes,np.float32)
        self.register_persistent("p")
        self.labs_accum = np.ones(classes,dtype=np.float32)
        self.register_persistent("labs_accum")
    
    def accumulate(self,labs):
        for l in labs:
            argmax = F.argmax(l)
            
    def forward(self,shape):
        smpl = D.Dirichlet(self.labs_accum).sample(0)[0]
        return D.Categorical(F.broadcast_to(smpl,shape))
"""

class UnifiedModel(Chain):
    def __init__(self,pseudo_dset=None,use_lm=False):
        super(UnifiedModel,self).__init__()
        with self.init_scope():
            self.classifier = net_chordrec.Classifier_RNN()
            self.generator = vae.ChromaVAE(pseudo_dset=pseudo_dset)
        transmatrix = transition_loop(C.N_VOCABULARY_TRIADS,C.SELF_TRANS_RATE).astype(np.float32)
        self.add_persistent("transmatrix",np.log(transmatrix))
        
        self.param_transition_a = 1.
        self.param_transition_b = 0.1
        
        self.beta_langmodel = 1
        self.update_steps = 20.0

        class_weight = np.ones(C.N_VOCABULARY_TRIADS,dtype=np.float32)
        class_weight[12:] = 2
        class_weight[24:] = 20
        class_weight[-1] = 2
        self.add_persistent("class_weight",class_weight)

    def update_beta(self):
        self.update_steps += 1
    def _gumbel_softmax_hard(self,x):
        y_soft = F.gumbel_softmax(x)
        index = F.argmax(y_soft,-1).data
        y_hard = self.xp.zeros_like(x.data)
        y_hard[self.xp.arange(len(index)),index] = 1
        y_hard = y_hard - y_soft.data + y_soft
        return y_hard
    def _hard_softmax_viterbi(self,x):
        x_filtered = F.sum(x[:-1,:,None] + self.transmatrix[None,:,:],axis=1) + x[1:,:]
        x_filtered = F.concat([x[:1,:],x_filtered],axis=0)
        y_soft = F.gumbel_softmax(x_filtered)
        #index = F.transpose_sequence(F.argmax_crf1d(self.transmatrix, F.transpose_sequence([F.log_softmax(x)]))[1])[0].data
        index = vtb(self.xp.asnumpy(y_soft.data.T),transition_loop(C.N_VOCABULARY_TRIADS,0.90).astype(np.float32))
        y_hard = self.xp.zeros_like(x.data)
        y_hard[self.xp.arange(len(index)),index] = 1
        y_hard = y_hard - y_soft.data + y_soft
        return y_hard
    def _hard_gumbel_forward(self,x):
        #x_logit = F.log_softmax(x,axis=-1)
        x_filtered = F.sum(x[:-1,:,None] + self.transmatrix[None,:,:],axis=1) + x[1:,:]
        x_filtered = F.concat([x[:1,:],x_filtered],axis=0)
        return self._gumbel_softmax_hard(x_filtered)

    def _update_transition_param(self,labs):
        labs_argmax = [l.data.argmax(axis=1) for l in labs]
        diff = [(l[:-1]-l[1:])==0 for l in labs_argmax]
        a = float(sum([d.sum() for d in diff]))
        b = sum([d.size for d in diff]) - a
        self.param_transition_b = b/a

    def get_transmatrix(self):
        phi = 1/(1+self.param_transition_b)
        mat = self.xp.identity(C.N_VOCABULARY_TRIADS,dtype=np.float32) * phi
        mat[mat==0] = (1-phi) / (C.N_VOCABULARY_TRIADS-1)
        return self.xp.log(mat)
    
    def _encode_onehot(self,y,n_category=C.N_VOCABULARY_TRIADS):
        xp = self.xp
        code = xp.zeros((len(y),n_category),dtype=xp.float32)
        code[xp.arange(code.shape[0]),y] = 1
        return code

    def _eval_markov_batch_manual(self,ps):
        seqlen = len(ps[0])
        ps_softmax = [F.softmax(p) for p in ps]
        score=0
        transmat = self.transmatrix
        if C.MARKOV_REGULARIZE:
            ps_trans = F.transpose_sequence(ps_softmax)
            alpha = Variable(self.xp.zeros(ps_trans[0].shape,dtype=self.xp.float32))     #alpha.shape = (batch,voc)
            for p in ps_trans[:-1]:
                alpha = F.sum((alpha[:,:,None] + transmat[None,:,:])*p[:,:,None], axis=1)
            score = -F.mean(F.sum(alpha*ps_trans[-1],axis=1)) / seqlen
        entropy = sum([F.mean(F.sum(F.log_softmax(p)*p_norm,axis=1)) for p,p_norm in zip(ps,ps_softmax)]) / len(ps)
        reporter.report({"loss_lang":score,
                         "entropy":-entropy},self)
        return (score + (C.VAE_WEIGHT_ENTROPY*entropy)) if C.MARKOV_REGULARIZE else entropy

    def _symbolize(self,labs,path):
        list_out = []
        idx_start = 0
        idx_cur = 1
        while idx_cur<len(path):
            while (idx_cur<len(path)) and (path[idx_start]==path[idx_cur]):
                idx_cur += 1
            list_out.append(F.sum(labs[idx_start:idx_cur],axis=0))
            idx_start = idx_cur

        out_concat = F.stack(list_out,axis=0)
        return F.gumbel_softmax(out_concat,axis=1)


    def loss_chordrec_only(self,xs,ys,aligns=None,xs_u=None):
        batchsize = len(xs)
        labs = self.classifier(xs)
        if aligns is None:
            ts = ys
        else:
            ts = [y[a] for y,a in zip(ys,aligns)]
        #cweight = self.xp.asarray(self.class_weight)
        loss = sum([F.softmax_cross_entropy(l,t,class_weight=None) for l,t in zip(labs,ts)]) / batchsize
        #loss += sum([-F.mean(F.sum(F.log_softmax(l)*F.softmax(l),axis=1)) for l in labs]) / batchsize

        accr = sum([F.accuracy(y,t) for y,t in zip(labs,ts)]) / batchsize
        perplexity  = sum([F.mean(1./F.softmax(l)[self.xp.arange(l.shape[0]),F.argmax(l,axis=1).data]) for l in labs]) / batchsize
        #reporter.report({"loss_gen":loss_generator},self)
        reporter.report({"perplex":perplexity},self)
        reporter.report({"loss_lab":loss},self)
        reporter.report({"accr":accr},self)
        return loss
    
    def loss_vae_only(self,xs,ys,aligns=None,return_zs=False):
        #loss_ncut = sum([self._soft_ncut_loss(f,p) for f,p in zip(xs,labs)]) / batchsize
        if aligns is None:
            ts = ys
        else:
            ts = [y[a] for y,a in zip(ys,aligns)]
        loss_generator,zs = self.generator(xs,[self._encode_onehot(l) for l in ts],return_zs=True)
        loss = loss_generator
        reporter.report({"loss":loss}, self)
        if return_zs:
            return loss,zs
        else:
            return loss   
     
    
    def loss_supervised(self,xs,ys,aligns=None):
        batchsize = len(xs)
        labs = self.classifier(xs)
        if aligns is None:
            ts = ys
        else:
            ts = [y[a] for y,a in zip(ys,aligns)]
        #cweight = self.xp.asarray(self.class_weight)
        loss_lab = sum([F.softmax_cross_entropy(l,t,class_weight=None)  for l,t in zip(labs,ts)]) / batchsize
        #loss_lab += sum([-F.mean(F.sum(F.log_softmax(l)*F.softmax(l),axis=1)) for l in labs]) / batchsize
        #loss_generator = self.generator(xs,ts)
        loss_generator = self.generator(xs,[self._encode_onehot(l) for l in ts],anneal_step=self.update_steps if C.VAE_ANNEAL_BETA else None)
        #loss_generator = self.generator(xs,[self._hard_gumbel_forward(l) for l in labs])
        #loss_generator = 0.0
        accr = sum([F.accuracy(y,t) for y,t in zip(labs,ts)]) / batchsize
        loss_mark = self._eval_markov_batch_manual(labs)
        #loss_jsd = sum([JSD(l[1:],l[:-1]) for l in [F.softmax(l) for l in labs]]) / batchsize
        reporter.report({"loss_gen_label":loss_generator},self)
        reporter.report({"loss_lab":loss_lab}, self)
        reporter.report({"accr":accr},self)
        return loss_lab+loss_generator+loss_mark
    
    def loss_unsupervised(self,xs,ys,aligns=None,return_zs=False):
        batchsize = len(xs)
        labs = self.classifier(xs)
        if aligns is None:
            ts = ys
        else:
            ts = [y[a] for y,a in zip(ys,aligns)]
        #loss_generator = self.generator(xs,ts)
        loss_generator,zs = self.generator(xs,[F.gumbel_softmax(l,axis=1) for l in labs],return_zs=True)
        accr = sum([F.accuracy(y,t) for y,t in zip(labs,ts)]) / batchsize
        reporter.report({"loss_gen_label":loss_generator},self)
        reporter.report({"accr":accr},self)
        if return_zs:
            return [loss_generator],zs
        else:
            return loss_generator,
    
    def loss_semisupervised(self,xs,xs_u,ys,aligns=None):
        batchsize = len(xs)
        labs = self.classifier(xs)
        labs_u = self.classifier(xs_u)
        if aligns is None:
            ts = ys
        else:
            ts = [y[a] for y,a in zip(ys,aligns)]
            
        loss_lab = sum([F.softmax_cross_entropy(l,t,class_weight=None) for l,t in zip(labs,ts)]) / batchsize
        loss_generator = self.generator(xs,[self._encode_onehot(l) for l in ts])
        #labs_sampled = [F.gumbel_softmax(l) for l in labs_u]
        labs_sampled = [self._hard_gumbel_forward(l) if C.MARKOV_REGULARIZE else self._gumbel_softmax_hard(l) for l in labs_u]
        loss_generator_u = self.generator(xs_u, labs_sampled,
                                          anneal_step=self.update_steps if C.VAE_ANNEAL_BETA else None)
        #loss_jsd = sum([JSD(l[1:],l[:-1]) for l in [F.softmax(l) for l in labs_u]]) / batchsize
        
        #self._update_transition_param(labs)
        self._update_transition_param(labs_u)
        labs_u.extend(labs)
        loss_mark = self._eval_markov_batch_manual(labs_u)
        #loss_jsd = sum([self._eval_markov(l) for l in labs_u]) / batchsize
        #loss_jsd = sum([self._eval_markov(l) for l in labs_u]) / batchsize
        #loss = loss_lab + 0.1*(loss_generator + loss_generator_u)
        accr = sum([F.accuracy(y,t) for y,t in zip(labs,ts)]) / batchsize
        perplexity_u = sum([F.mean(1./F.softmax(l)[self.xp.arange(l.shape[0]),F.argmax(l,axis=1).data]) for l in labs_u]) / batchsize
        perplexity = sum([F.mean(1./F.softmax(l)[self.xp.arange(l.shape[0]),F.argmax(l,axis=1).data]) for l in labs]) / batchsize
        
        reporter.report({"perplex":perplexity,
        "perplex_u":perplexity_u,
        "trans_param_b":self.param_transition_b,
        "loss_lab":loss_lab,
        "loss_gen_label":loss_generator,
        "loss_gen_unlabel":loss_generator_u,
        "accr":accr},self)
        return loss_lab+loss_generator+loss_generator_u+loss_mark

    def loss_semisupervised_separate(self,xs,xs_u,ys,aligns=None):
        batchsize = len(xs)
        labs = self.classifier(xs)
        labs_u = self.classifier(xs_u)
        if aligns is None:
            ts = ys
        else:
            ts = [y[a] for y,a in zip(ys,aligns)]
            
        #cweight = self.xp.asarray(self.class_weight)
        loss_lab = sum([F.softmax_cross_entropy(l,t,class_weight=None) for l,t in zip(labs,ts)]) / batchsize
        loss_generator = self.generator(xs,[self._encode_onehot(l) for l in ts])
        #labs_sampled = [F.gumbel_softmax(l) for l in labs_u]
        labs_sampled = [self._hard_gumbel_forward(l) if C.MARKOV_REGULARIZE else self._gumbel_softmax_hard(l) for l in labs_u]
        loss_generator_u = self.generator(xs_u, labs_sampled,
                                          anneal_step=self.update_steps if C.VAE_ANNEAL_BETA else None)
        #loss_jsd = sum([JSD(l[1:],l[:-1]) for l in [F.softmax(l) for l in labs_u]]) / batchsize
        
        #self._update_transition_param(labs)
        self._update_transition_param(labs_u)
        labs_u.extend(labs)
        loss_mark = self._eval_markov_batch_manual(labs_u)
        #loss_jsd = sum([self._eval_markov(l) for l in labs_u]) / batchsize
        #loss_jsd = sum([self._eval_markov(l) for l in labs_u]) / batchsize
        #loss = loss_lab + 0.1*(loss_generator + loss_generator_u)
        accr = sum([F.accuracy(y,t) for y,t in zip(labs,ts)]) / batchsize
        perplexity_u = sum([F.mean(1./F.softmax(l)[self.xp.arange(l.shape[0]),F.argmax(l,axis=1).data]) for l in labs_u]) / batchsize
        perplexity = sum([F.mean(1./F.softmax(l)[self.xp.arange(l.shape[0]),F.argmax(l,axis=1).data]) for l in labs]) / batchsize
        
        reporter.report({"perplex":perplexity,
        "perplex_u":perplexity_u,
        "trans_param_b":self.param_transition_b,
        "loss_lab":loss_lab,
        "loss_gen_label":loss_generator,
        "loss_gen_unlabel":loss_generator_u,
        "accr":accr},self)
        return loss_generator_u, loss_lab+loss_generator+loss_mark
    
    def loss_semi_unsuploss(self,xs,xs_u,ys,aligns=None):
        batchsize = len(xs)
        labs = self.classifier(xs)
        if len(xs_u)>0:
            labs_u = self.classifier(xs_u)
        else:
            labs_u = []
        if aligns is None:
            ts = ys
        else:
            ts = [y[a] for y,a in zip(ys,aligns)]
        xs_u.extend(xs)
        labs_u.extend(labs)
        #cweight = self.xp.asarray(self.class_weight)
        loss_lab = sum([F.softmax_cross_entropy(l,t,class_weight=None) for l,t in zip(labs,ts)]) / batchsize
        #loss_generator = self.generator(xs,[self._encode_onehot(l) for l in ts])
        #labs_sampled = [F.gumbel_softmax(l) for l in labs_u]
        labs_sampled = [self._hard_gumbel_forward(l) if C.MARKOV_REGULARIZE else self._gumbel_softmax_hard(l) for l in labs_u]
        loss_generator_u = self.generator(xs_u, labs_sampled,
                                          anneal_step=self.update_steps if C.VAE_ANNEAL_BETA else None)
        #loss_jsd = sum([JSD(l[1:],l[:-1]) for l in [F.softmax(l) for l in labs_u]]) / batchsize
        
        #self._update_transition_param(labs)
        self._update_transition_param(labs_u)
        
        loss_mark = self._eval_markov_batch_manual(labs_u)
        #loss_jsd = sum([self._eval_markov(l) for l in labs_u]) / batchsize
        #loss_jsd = sum([self._eval_markov(l) for l in labs_u]) / batchsize
        #loss = loss_lab + 0.1*(loss_generator + loss_generator_u)
        accr = sum([F.accuracy(y,t) for y,t in zip(labs,ts)]) / batchsize
        #perplexity_u = sum([F.mean(1./F.softmax(l)[self.xp.arange(l.shape[0]),F.argmax(l,axis=1).data]) for l in labs_u]) / batchsize
        #perplexity = sum([F.mean(1./F.softmax(l)[self.xp.arange(l.shape[0]),F.argmax(l,axis=1).data]) for l in labs]) / batchsize
        
        reporter.report({
        "trans_param_b":self.param_transition_b,
        "loss_lab":loss_lab,
        "loss_gen_unlabel":loss_generator_u,
        "accr":accr},self)
        return loss_lab+loss_generator_u+loss_mark
    
    
    def loss_discrim(self,xs,ys):
        loss = self.generator.loss_discrim(xs,[self._encode_onehot(l) for l in ys])
        return loss

    def likelihood(self,x,y,align):
        pred = self.classifier([x])
        pred = [F.gumbel_softmax(l) for l in pred]
        loss_gumbel = self.generator.reconstr_loss([x],pred)
        loss_supervise = self.generator.reconstr_loss([x],[U.encode_onehot(y)[align]])
        return loss_supervise,loss_gumbel,F.argmax(pred[0],axis=1).data
        
    
    def eval_chordrec(self,xs,ys,aligns):
        batchsize = len(xs)
        labs_estimate = self.classifier(xs)
        accr = sum([F.accuracy(x,y[a]) for x,y,a in zip(labs_estimate,ys,aligns)]) / batchsize
        reporter.report({"accr":accr}, self)

    def eval_generator_shift(self,xs,ys):
        batchsize = len(xs)
        ys_onehot = [self._encode_onehot(y) for y in ys]
        with using_config('enable_backprop', False):
            for shift in range(1,12):
                reconstr,shifted_xs = self.generator.reconstr_dist(xs,ys_onehot,shift=shift)
            logprob = sum([F.mean(F.sum(dist.log_prob(x),axis=-1)) for dist,x in zip(reconstr,shifted_xs)])/batchsize
        reporter.report({"log_prob":logprob},self)

    def reconst(self,xs):
        labs = self.classifier([xs])[0]
        labs_onehot = self._gumbel_softmax_hard(labs)
        ret = self.generator.reconstr_dist([xs],[labs_onehot],None)[0]
        return ret, labs.data.argmax(-1)

    def _viterbi_hierarchy(self,p_s,est_beat):
        n_steps, n_states = p_s.shape[0],p_s.shape[1]
        states = np.zeros(n_steps,dtype=int)
        values = np.zeros((n_steps, n_states), dtype=float)
        ptr = np.zeros((n_steps, n_states),dtype=int)
        
        log_p_init = np.log(np.full(n_states, 1./n_states))
        
        values[0] = p_s[0] + log_p_init
        
        #transmat_s_h = self.transmatrix_key_chord[est_key,:,:].cpu().numpy() #(T,voc_s,voc_s)
        #transmat_s_h = np.log(chord.key_chord_norm[est_key,:,:])
        #transmat_s_h[:] = 0
        transmat_beat_chord = np.stack([transition_loop(n_states,0.9999).astype(np.float32),
                            transition_loop(n_states,0.1).astype(np.float32)])
        transmat_s_r = np.log(transmat_beat_chord[est_beat,:,:])
        transmat_s = np.log(transition_loop(n_states,0.9))
        for t in range(1, n_steps):
            trans_out = values[t-1] + transmat_s_r[t,:,:] + transmat_s
            for j in range(n_states):
                ptr[t,j] = np.argmax(trans_out[j])
                values[t,j] = p_s[t,j] + trans_out[j, ptr[t,j]]
        
        states[-1] = np.argmax(values[-1])
        for t in range(n_steps-2,-1,-1):
            states[t] = ptr[t+1,states[t+1]]
        return states
    
    def estimate(self,x):
        labs = self.classifier([x])[0]
        if C.POSTFILTER:
            labs_argmax = F.transpose_sequence(F.argmax_crf1d(self.transmatrix, F.transpose_sequence([F.log_softmax(labs,axis=-1)]))[1])[0].data
        else:
            labs_argmax = F.argmax(labs,axis=1).data
        return labs_argmax
    
    def estimate_beatsync(self,x,beat_pos):
        labs = F.log_softmax(self.classifier([x])[0],axis=-1).data.get()
        beatlab = np.zeros(labs.shape[0],dtype=int)
        beatlab[beat_pos[beat_pos<labs.shape[0]]] = 1
        path = self._viterbi_hierarchy(labs,beatlab)
        return path
    
    def estimate_transrate(self,x):
        labs = self.classifier([x])[0]
        labs_argmax = F.argmax(labs,axis=1).data
        labs_argmax_diff = labs_argmax[1:] - labs_argmax[:-1]
        trans_rate = labs_argmax.size / float(labs_argmax_diff.nonzero()[0].size + 1)
        return trans_rate
        
    
    def estimate_entropy(self,x):
        outs = self.classifier([x])[0]
        entropy = -F.mean(F.sum(F.log_softmax(outs)*F.softmax(outs),-1)).data
        return entropy

    def save(self,path):
        serializers.save_npz(path,self)
    def load(self,path):
        serializers.load_npz(path,self,ignore_names=["transmatrix","generator/label_shifter",
                                                     "generator/chroma_shifter",
                                                     "generator/encoder/chroma_shifter",
                                                     "cat_prior/p"])


