import cPickle, itertools, numpy as np, sys
from os.path import join
from scipy.stats import entropy

mapper = ['A','C','G','T']
re_mapper = {'A':0,'C':1,'G':2,'T':3}

def data2prob(data_ori,idx_mapper):
    # Calcualte the emperical kmer distribution from a generated dataset
    out = np.zeros(len(idx_mapper.keys()))
    data = data_ori.squeeze().swapaxes(1,2)
    for x in data:
        t_data = [ mapper[y.argmax()]   for y in x]
        out[idx_mapper[''.join(t_data)]] += 1
    return out/sum(out)

def kl_compare(rundir, motif_file, max_epoch, seqlen):
    # All posible kmers
    candidate = [''.join(p) for p in itertools.product(mapper, repeat=seqlen)]

    # Map each kmer to its index in the list
    idx_mapper = dict()
    for idx,x in enumerate(candidate):
        idx_mapper[x] = idx

    # Read the motif
    with open(motif_file) as f:
        f.readline()
        motif_mat = [map(float,x.split()) for x in f]

    # Calculate the expected probability of each kmer
    design_p = np.zeros(len(candidate))
    for idx,x in enumerate(candidate):
        t_p = 1.0
        for cidx, c in enumerate(list(x)):
            t_p *= motif_mat[cidx][re_mapper[c]]
        design_p[idx] = t_p

    # For each epoch, calculate the emperical probability of each kmer
    # and compare with the expectation with KL divergence
    kl_per_epoch = []
    #data = cPickle.load(open(join(rundir,'plot_epoch_{0:03}_generated.pkl'.format(49))))
    #pred_p = data2prob(data, idx_mapper)
    #return entropy(pred_p, design_p)
    for epoch in range(max_epoch):
        #print 'epoch ',epoch
        data = cPickle.load(open(join(rundir,'plot_epoch_{0:03}_generated.pkl'.format(epoch))))
        pred_p = data2prob(data, idx_mapper)
        kl_per_epoch.append(entropy(pred_p, design_p))
    return kl_per_epoch

motif_file = sys.argv[1]
max_epoch = int(sys.argv[2])
seqlen = int(sys.argv[3])
rundir = sys.argv[4]

kl = kl_compare(rundir, motif_file, max_epoch, seqlen)
cPickle.dump(kl,open(join(rundir,'kl.pkl'),'wb'))
