from os.path import join,exists,realpath,dirname
from os import makedirs,listdir
import numpy as np, sys

def readMotif(dfile):
    with open(dfile) as f:
        f.readline()
        return [map(float,x.split()) for x in f]

def gen_motif_instance(pwm):
    return [ mydict[np.argmax(np.random.multinomial(1,x))] for x in pwm]

def sample_loc(_range, _len):
    return np.random.randint(_range[1] - _range[0] + 1 - _len + 1) + _range[0]

def grammar_spikein(seq,grammars):
    _lrange, _rrange, t_pwm = grammars
    t_motif =  gen_motif_instance(t_pwm)
    _left = sample_loc([_lrange,_rrange], len(t_motif))
    _right = _left + len(t_motif)
    seq[_left:_right] = t_motif
    return seq

mydict = ['A','C','G','T']
def rand_seq(num):
    return [ mydict[x] for x in np.random.randint(4, size=num)]

seq_num = int(sys.argv[1])
seqlen = int(sys.argv[2])
cwd = dirname(realpath(__file__))
datadir = sys.argv[3]
motif_file = sys.argv[4]

if not exists(datadir):
    makedirs(datadir)

motifs = (0,seqlen-1,readMotif(motif_file))

with open(join(datadir,'data.tsv'),'w') as fseq, open(join(datadir,'data.label'),'w') as flabel:
    for cnt in range(seq_num):
        t_label = np.random.randint(len(motifs))
        t_seq = grammar_spikein(rand_seq(seqlen),motifs)
        fseq.write('>seq-{}\t{}\n'.format(str(cnt),''.join(t_seq)))
        flabel.write('%s\n' % str(t_label))
