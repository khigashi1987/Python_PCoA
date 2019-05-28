import numpy as np
import math 
 
def convert2freq(seq):
    sum_all = float(sum(seq))
    return [ float(x) / sum_all for x in seq ]
 
 
def entropy(seq):
    pdf = convert2freq(seq)
    H = 0.
    for p in pdf:
        if p == 0.:
            continue
        H += - p * math.log(p,2)
 
    return H
 
def KLdivergence(seqTarget, seqBackground):
    if len(seqTarget) != len(seqBackground):
        print('error: dimensions of two sequences must be same')
        return None
    freqTarget = convert2freq(seqTarget)
    freqBackground = convert2freq(seqBackground)
    KLD = 0.
    for (p,q) in zip(freqTarget,freqBackground):
        if p == 0. or q == 0.:
            continue
        KLD += p * math.log(p / q, 2)
    return KLD
 
def JSdivergence(seq1, seq2, weights=(0.5,0.5)):
    if len(seq1) != len(seq2):
        print('error: dimensions of two sequences must be same')
        return None
    freq1 = convert2freq(seq1)
    freq2 = convert2freq(seq2)
    averageFreq = []
    for (f1,f2) in zip(freq1, freq2):
        averageFreq.append( weights[0] * f1 + weights[1] * f2)
    JSD = entropy(averageFreq) - weights[0] * entropy(freq1) - weights[1] * entropy(freq2)
    return JSD
