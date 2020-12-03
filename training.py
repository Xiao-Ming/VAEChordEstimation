#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:40:57 2019

@author: wuyiming
"""

import numpy as np
import vae
import const as C
import dataset as D
import util as U


import chainer
from chainer import training,optimizers,optimizer_hooks,iterators,cuda,report
from chainer.dataset import to_device

import net_generative

def convert_chordlm_onehot(batch,device):
    list_label= [to_device(device,U.encode_onehot(l)) for l in batch]
    return {"xs":list_label}

def convert_target_aligns(batch,device):
    list_feat = [to_device(device,feat) for feat,_,_ in batch]
    list_labs = [to_device(device,lab) for _,lab,_ in batch]
    list_aligns = [to_device(device,align) for _,_,align in batch]
    #list_mfcc = [to_device(device,mfcc) for _,_,_,mfcc in batch]
    
    return {'xs':list_feat,
            'aligns':list_aligns,
            'ys':list_labs}
    
def convert_onehot_aligns(batch,device):
    list_feat = [to_device(device,feat) for feat,_,_ in batch]
    list_labs = [to_device(device,U.encode_onehot(lab)) for _,lab,_ in batch]
    list_aligns = [to_device(device,align) for _,_,align in batch]
    #list_mfcc = [to_device(device,mfcc) for _,_,_,mfcc in batch]
    
    return {'xs':list_feat,
            'aligns':list_aligns,
            'ys':list_labs}

def convert_target_aligns_semi(batch,device):
    argsort = np.argsort([feat.shape[0] for feat,_,_,_ in batch])[::-1]
    list_feat = [to_device(device,batch[i][0]) for i in argsort]
    argsort_u = np.argsort([feat.shape[0] for _,feat,_,_ in batch])[::-1]
    list_feat_u = [to_device(device,batch[i][1]) for i in argsort_u]
    list_labs = [to_device(device,batch[i][2][batch[i][3]]) for i in argsort]
    #list_aligns = [to_device(device,align) for _,_,_,align in batch]
    #list_mfcc = [to_device(device,mfcc) for _,_,_,mfcc in batch]
    
    return {'xs':list_feat,
            'xs_u':list_feat_u,
            'ys':list_labs}

def convert_target_aligns_fixlen(batch,device):
    list_feat = [feat for feat,_,_ in batch]
    list_labs = [lab for _,lab,_ in batch]
    list_aligns = [align for _,_,align in batch]
    length_min = min([C.SEQLEN_TRAINING,min([f.shape[0] for f in list_feat])])
    length_max = min([f.shape[0] for f in list_feat])+1
    #length = np.random.randint(length_min,length_max)
    length = length_min
    list_st = [np.random.randint(f.shape[0]-length+1) for f in list_feat]
    list_feat = [to_device(device,f[s:s+length]) for f,s in zip(list_feat,list_st)]
    list_aligns = [a[s:s+length] for a,s in zip(list_aligns,list_st)]
    ts = [to_device(device,l[a]) for l,a in zip(list_labs,list_aligns)]
    
    return {'xs':list_feat,
            'ys':ts}  

from itertools import chain as iterchain

def convert_target_aligns_semi_fixlen(batch,device):
    list_feat = [feat for feat,_,_,_ in batch]
    list_feat_u = [feat for _,feat,_,_ in batch]
    #list_feat_u = list(iterchain.from_iterable(list_feat_u))
    list_labs = [lab for _,_,lab,_ in batch]
    list_aligns = [align for _,_,_,align in batch]
    length_min = min([C.SEQLEN_TRAINING,min([f.shape[0] for f in list_feat]),min([f.shape[0] for f in list_feat_u])])
    #length_max = min([min([f.shape[0] for f in list_feat]),min([f.shape[0] for f in list_feat_u])])+1
    #length = np.random.randint(length_min,length_max)
    length = length_min
    list_st = [np.random.randint(f.shape[0]-length+1) for f in list_feat]
    list_st_u = [np.random.randint(f.shape[0]-length+1) for f in list_feat_u]
    
    list_feat = [to_device(device,f[s:s+length]) for f,s in zip(list_feat,list_st)]
    list_feat_u = [to_device(device,f[s:s+length]) for f,s in zip(list_feat_u,list_st_u)]
    list_aligns = [a[s:s+length] for a,s in zip(list_aligns,list_st)]
    ts = [to_device(device,l[a]) for l,a in zip(list_labs,list_aligns)]
    
    return {'xs':list_feat,
            'xs_u':list_feat_u,
            'ys':ts}    
    

def convert_batch_semi(batch,device):
    xs = np.stack([x for x,_,_ in batch])
    ts = np.array([y for _,y,_ in batch],dtype=np.int32)
    xs_u = np.stack([x for _,_,x in batch])
    return {"xs":to_device(device,xs),
            "xs_u":to_device(device,xs_u),
            "ts":to_device(device,ts)}


    
def update_beta(trainer):
    trainer.updater.get_optimizer("main").target.update_beta()


def TrainGenerativeModelSemiSupervise(idx_train, idx_eval,idx_unlabelled, device, epoch=50, log_interval=1, resume=None, save_model="", log_name="test"):
    if device>0:
        cuda.cuda.Device(device).use()
    print("Loading data...")
    train_set = D.ChordDatasetSemisupervised(idx_train,idx_unlabelled,rand_shift=True)
    #train_set = D.ChordDataset(idx_train,rand_shift=True)
    #eval_set = D.ChordDataset(idx_eval,rand_shift=False)
    #eval_set2 = D.ChordDataset(np.load("idx_non_billboard.npy")+320,rand_shift=False)
    
    iter_train = iterators.SerialIterator(train_set,C.N_BATCH,repeat=True)
    #iter_eval = iterators.SerialIterator(eval_set,C.N_BATCH,repeat=False)
    print("Preparing model...")
    #model = net_generative.GenerativeChordnet(pseudo_dset = D.ChordDataset(idx_eval[:16],rand_shift=True))
    model = net_generative.UnifiedModel(use_lm=C.LANGUAGEMODEL,pseudo_dset=D.ChordDataset(idx_eval[:16],rand_shift=True) if C.PSEUDO_PRIOR else None)
    print("model:"+model.__class__.__name__)

    if resume is not None:
        model.load(resume)

    """
    model.cat_prior.p = train_set.chord_distribution()
    print("label prior:")
    print(model.cat_prior.p)
    """
    opt = optimizers.Adam()
    opt.setup(model)
    opt.add_hook(optimizer_hooks.GradientClipping(5))
    #opt.add_hook(optimizer_hooks.WeightDecay(0.0001))
    #model.generator.disable_update()
    
    updater = training.StandardUpdater(iter_train,opt,converter=convert_target_aligns_semi_fixlen,\
                                          device=device,loss_func=model.loss_semisupervised) if C.GENERATOR_SEMI else \
            custom_updater.DoubleUpdater(iter_train,opt,converter=convert_target_aligns_semi_fixlen,
                                   device=device,loss_func=model.loss_semisupervised_separate)
    
    #updater = custom_updater.AdVAEUpdater(iter_train,{"main":opt_cf,"d":opt},converter=convert_target_aligns_semi,\
    #                                      device=device,loss_func={"main":model_classifier.loss_chordrec_only,"d":model.loss_semisupervised})
    trainer = training.Trainer(updater,(epoch,"epoch"))
    log_keys = ['epoch', 
                   'iteration',
                   'main/loss_lab',
                   'main/entropy',
                   'main/loss_lang',
                   'main/trans_param_b',
                   'main/accr',
                   'main/generator/kl',
                   'main/generator/beta',
                   'main/loss_gen_label',
                   'main/loss_gen_unlabel',
                   'elapsed_time']
    
    #trainer.extend(training.extensions.Evaluator(iter_eval,model,device=device,converter=convert_target_aligns,eval_func=model.loss_chordrec_only))
    trainer.extend(training.extensions.ExponentialShift("alpha",0.99),trigger=(1,"epoch"))
    if C.VAE_ANNEAL_BETA:
        trainer.extend(update_beta,trigger=(1,"epoch"))
    trainer.extend(training.extensions.LogReport(trigger=(log_interval, 'epoch'),log_name=log_name))
    trainer.extend(training.extensions.PrintReport(log_keys), trigger=(log_interval, 'epoch'))
    
    print("Training epoch start")
    trainer.run()
    
    model.save(save_model)





    
def TrainGenerativeModelSupervised(idx_train, idx_eval, device, epoch=50, log_interval=1, resume=None, save_model="",log_name="test"):
    if device>0:
        cuda.cuda.Device(device).use()
    print("Loading data...")
    train_set = D.ChordDataset(idx_train,rand_shift=True)
    #eval_set = D.ChordDataset(idx_eval,rand_shift=False)
    
    iter_train = iterators.SerialIterator(train_set,C.N_BATCH,repeat=True)
    #iter_eval = iterators.SerialIterator(eval_set,C.N_BATCH,repeat=False)
    print("Preparing model...")
    model = net_generative.UnifiedModel(use_lm=C.LANGUAGEMODEL, pseudo_dset = D.ChordDataset(idx_eval[:32],rand_shift=True) if C.PSEUDO_PRIOR else None)
    print("model:"+model.__class__.__name__)
    #model = net_generative.LadderLSTM()
    if resume is not None:
        model.load(resume)
    
    opt = optimizers.Adam()
    opt.setup(model)
    opt.add_hook(optimizer_hooks.GradientClipping(5))
    #opt.add_hook(optimizer_hooks.WeightDecay(0.0001))

    #updater_supervised = custom_updater.AdVAEUpdater(iter_train,{'main':opt,'d':opt_d},converter=convert_target_aligns,\
    #                    device=device,loss_func={'main':model.loss_supervised,'d':model_discrim})    
    updater_supervised = training.StandardUpdater(iter_train,opt,converter=convert_target_aligns_fixlen,device=device,loss_func=model.loss_chordrec_only)
    trainer_supervised = training.Trainer(updater_supervised,(epoch,"epoch"))

    
    #trainer_supervised.extend(training.extensions.Evaluator(iter_eval,model,device=device,converter=convert_target_aligns,eval_func=model.loss_chordrec_only))
    trainer_supervised.extend(training.extensions.ExponentialShift("alpha",0.99),trigger=(1,"epoch"))
    if C.VAE_ANNEAL_BETA:
        trainer_supervised.extend(update_beta,trigger=(1,"epoch"))
    trainer_supervised.extend(training.extensions.LogReport(trigger=(log_interval, 'epoch'),log_name=log_name))
    trainer_supervised.extend(training.extensions.PrintReport(['epoch', 
                                           'iteration', 
                                           'main/loss',
                                           'main/loss_lab',
                                           'main/loss_gen_label',
                                           'main/accr',
                                           'main/accr_bass',
                                           'main/accr_seventh',
                                           'main/generator/beta',
                                           'validation/main/loss_lab',
                                           'validation/main/accr',
                                           'elapsed_time']), \
                                            trigger=(log_interval, 'epoch'))
    print("Training epoch start")
    trainer_supervised.run()
    
    model.save(save_model)


 
    #trainer_supervised.extend(training.extensions.Evaluator(iter_eval,model,device=device,converter=convert_target_aligns,eval_func=model.loss_chordrec_only))
    if C.VAE_ANNEAL_BETA:
        trainer_supervised.extend(update_beta,trigger=(1,"epoch"))
    trainer_supervised.extend(training.extensions.LogReport(trigger=(log_interval, 'epoch'),log_name=log_name2))
    trainer_supervised.extend(training.extensions.PrintReport(['epoch', 
                                           'iteration', 
                                           'main/loss',
                                           'main/loss_lab',
                                           'main/loss_gen_label',
                                           'main/generator/beta',
                                           'main/accr',
                                           'main/accr_bass',
                                           'main/accr_seventh',
                                           'validation/main/loss_lab',
                                           'validation/main/accr',
                                           'elapsed_time']), \
                                            trigger=(log_interval, 'epoch'))
    trainer_supervised.run()
    
    model.save(save_model2)
    
"""    
def TrainGenerativeModelUnsupervised(idx_train, idx_eval, device, epoch=50, log_interval=1, resume=None, save_model="",log_name="test"):
    if device>0:
        cuda.cuda.Device(device).use()
    print("Loading data...")
    train_set = D.ChordDataset(idx_train,rand_shift=True)
    eval_set = D.ChordDataset(idx_eval,rand_shift=False)
    
    iter_train = iterators.SerialIterator(train_set,C.N_BATCH,repeat=True)
    iter_eval = iterators.SerialIterator(eval_set,C.N_BATCH,repeat=False)
    print("Preparing model...")
    model = net_generative.UnifiedModel()
    #model = net_generative.LadderLSTM()
    if resume is not None:
        model.load(resume)
    
    opt = optimizers.Adam()
    opt.setup(model)
    opt.add_hook(optimizer_hooks.GradientClipping(10))
    #opt.add_hook(optimizer_hooks.WeightDecay(0.0001))

    #updater_supervised = custom_updater.AdVAEUpdater(iter_train,{'main':opt,'d':opt_d},converter=convert_target_aligns,\
    #                    device=device,loss_func={'main':model.loss_supervised,'d':model_discrim})    
    updater_supervised = custom_updater.DoubleUpdater(iter_train,opt,converter=convert_target_aligns_fixlen,device=device,loss_func=model.loss_unsupervised)
    trainer_supervised = training.Trainer(updater_supervised,(epoch,"epoch"))

    
    trainer_supervised.extend(training.extensions.Evaluator(iter_eval,model,device=device,converter=convert_target_aligns,eval_func=model.loss_chordrec_only))
    trainer_supervised.extend(training.extensions.LogReport(trigger=(log_interval, 'epoch'),log_name=log_name))
    trainer_supervised.extend(training.extensions.PrintReport(['epoch', 
                                           'iteration', 
                                           'main/loss',
                                           'main/loss_lab',
                                           'main/loss_gen_label',
                                           'main/accr',
                                           'validation/main/loss_lab',
                                           'validation/main/accr',
                                           'elapsed_time']), \
                                            trigger=(log_interval, 'epoch'))
    print("Training epoch start")
    with chainer.using_config("debug",False):
        trainer_supervised.run()
    
    model.save(save_model)
"""


import chord
def Estimate(idx,load_model,device,model_class=0,verbose=False):
    def idx2sec(i):
        return i * C.H / float(C.SR)
    
    #classes = [net_generative.UnifiedModel]
    dset = D.ChordDataset(idx)
    
    #model = NA.AttentionNMFClassifier()
    model = net_generative.UnifiedModel()
    model.load(load_model)
    #model.hmm.reset_priors()
    if device>=0:
        cuda.cuda.Device(device).use()
        model.to_gpu(device)
    
    with chainer.using_config('train', False):
        chainer.config.enable_backprop = False
        for i_data in range(len(idx)):
            labfile = open(U.path_estimated_lab(dset.list_labfile[i_data]),"w")
            if verbose:
                print("Estimating: %s" % U.path_estimated_lab(dset.list_labfile[i_data]))
            
            feat,_,_ = dset[i_data]
            feat = to_device(device,feat)
            estimated_triad = model.estimate(feat)
            cur_triad = int(estimated_triad[0])
            cur_frame = 0
            for i_frame in range(len(estimated_triad)):
                if estimated_triad[i_frame]==cur_triad:
                    continue
                #cur_seventh = np.argmax(estimated_sevenths_prob[cur_frame:i_frame,:].sum(axis=0))
                sign = chord.id2signature(cur_triad)
                line = "%.4f %.4f %s\n" % (idx2sec(cur_frame),idx2sec(i_frame),sign)
                labfile.write(line)
                cur_frame = i_frame
                cur_triad = int(estimated_triad[i_frame])
            
            if cur_frame<i_frame:
                sign = chord.id2signature(cur_triad)
                line = "%.4f %.4f %s" % (idx2sec(cur_frame),idx2sec(i_frame),sign)
                labfile.write(line)
            
            labfile.close()
    
        chainer.config.enable_backprop = True

def Estimate_transrate(idx,load_model,device):
    def idx2sec(i):
        return i * C.H / float(C.SR)
    
    #classes = [net_generative.UnifiedModel]
    dset = D.ChordDataset(idx)
    
    #model = NA.AttentionNMFClassifier()
    model = net_generative.UnifiedModel()
    model.load(load_model)
    #model.hmm.reset_priors()
    if device>=0:
        cuda.cuda.Device(device).use()
        model.to_gpu(device)
    
    list_transrate = []
    with chainer.using_config('train', False):
        chainer.config.enable_backprop = False
        for i_data in range(len(idx)):
            feat,_,_ = dset[i_data]
            feat = to_device(device,feat)
            trans_rate = model.estimate_transrate(feat)
            list_transrate.append(trans_rate)
    
        chainer.config.enable_backprop = True
    
    return list_transrate
   

import librosa.core
import librosa.beat
from librosa.util import find_files
from madmom.features.beats import RNNBeatProcessor, CRFBeatDetectionProcessor

def Estimate_beatsync(idx,load_model,device,model_class=0,verbose=False):
    def idx2sec(i):
        return i * C.H / float(C.SR)
    def round2beatpos(i,beatpos):
        sec = idx2sec(i)
        return beatpos[np.argmin(np.abs(beatpos - sec))]
    
    #classes = [net_generative.UnifiedModel]
    dset = D.ChordDataset(idx)
    
    #model = NA.AttentionNMFClassifier()
    model = net_generative.UnifiedModel()
    model.load(load_model)
    #model.hmm.reset_priors()
    if device>=0:
        cuda.cuda.Device(device).use()
        model.to_gpu(device)
        
    beat_proc = CRFBeatDetectionProcessor(fps=100)
    
    with chainer.using_config('train', False):
        chainer.config.enable_backprop = False
        for i_data in range(len(idx)):
            labfile = open(U.path_estimated_lab(dset.list_labfile[i_data]),"w")
            print(dset.list_labfile[i_data])
            if verbose:
                print("Estimating: %s" % U.path_estimated_lab(dset.list_labfile[i_data]))
            audiofile = find_files(C.PATH_AUDIO)[idx[i_data]]
            print(audiofile)
            #audio, _ = librosa.core.load(audiofile,sr=C.SR)
            #_, beatpos = librosa.beat.beat_track(y=audio,sr=C.SR,hop_length=C.H,units="frames")
            act_beat = RNNBeatProcessor()(audiofile)
            beatpos_sec = beat_proc(act_beat)
            beatpos = np.round(beatpos_sec * C.SR / C.H).astype(int)
            feat,_,_ = dset[i_data]
            feat = to_device(device,feat)
            estimated_triad = model.estimate_beatsync(feat,beatpos)
            cur_triad = int(estimated_triad[0])
            cur_frame = 0
            for i_frame in range(len(estimated_triad)):
                if estimated_triad[i_frame]==cur_triad:
                    continue
                #cur_seventh = np.argmax(estimated_sevenths_prob[cur_frame:i_frame,:].sum(axis=0))
                sign = chord.id2signature(cur_triad)
                line = "%.4f %.4f %s\n" % (round2beatpos(cur_frame,beatpos_sec),round2beatpos(i_frame,beatpos_sec),sign)
                labfile.write(line)
                cur_frame = i_frame
                cur_triad = int(estimated_triad[i_frame])
            
            if cur_frame<i_frame:
                sign = chord.id2signature(cur_triad)
                line = "%.4f %.4f %s" % (round2beatpos(cur_frame,beatpos_sec),round2beatpos(i_frame,beatpos_sec),sign)
                labfile.write(line)
            
            labfile.close()
    
        chainer.config.enable_backprop = True

def Estimate_rawaudio(list_audiofile,list_labfile,load_model,device,model_class=0,verbose=False):
    def idx2sec(i):
        return i * C.H / float(C.SR)
    
    #classes = [net_generative.UnifiedModel]
    dset = D.ChordDataset(idx)
    
    #model = NA.AttentionNMFClassifier()
    
    model = net_generative.UnifiedModel()
    model.load(load_model)
    #model.hmm.reset_priors()
    if device>=0:
        cuda.cuda.Device(device).use()
        model.to_gpu(device)
    
    with chainer.using_config('train', False):
        chainer.config.enable_backprop = False
        for audiofile,labfile in zip(list_audiofile,list_labfile):
            y, sr = librosa.core.load(audiofile,sr=44100)
            if verbose:
                print("Estimating: %s" % audiofile)
            
            feat,_,_ = dset[i_data]
            feat = to_device(device,feat)
            estimated_triad = model.estimate(feat)
            cur_triad = int(estimated_triad[0])
            cur_frame = 0
            for i_frame in range(len(estimated_triad)):
                if estimated_triad[i_frame]==cur_triad:
                    continue
                #cur_seventh = np.argmax(estimated_sevenths_prob[cur_frame:i_frame,:].sum(axis=0))
                sign = chord.id2signature(cur_triad)
                line = "%.4f %.4f %s\n" % (idx2sec(cur_frame),idx2sec(i_frame),sign)
                labfile.write(line)
                cur_frame = i_frame
                cur_triad = int(estimated_triad[i_frame])
            
            if cur_frame<i_frame:
                sign = chord.id2signature(cur_triad)
                line = "%.4f %.4f %s" % (idx2sec(cur_frame),idx2sec(i_frame),sign)
                labfile.write(line)
            
            labfile.close()
    
        chainer.config.enable_backprop = True    

def Accum_Entropy(idx,load_model,device):
    dset = D.ChordDataset(idx)
    model = net_generative.UnifiedModel()
    model.load(load_model)
    #model.hmm.reset_priors()
    list_entropy = []
    if device>=0:
        cuda.cuda.Device(device).use()
        model.to_gpu(device)
    with chainer.using_config('train', False):
        chainer.config.enable_backprop = False
        for i_data in range(len(idx)):
            feat,_,_ = dset[i_data]
            feat = to_device(device,feat)
            entropy = model.estimate_entropy(feat)
            list_entropy.append(entropy)
    return np.mean(entropy)


