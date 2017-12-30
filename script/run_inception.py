from os import sys
from os.path import join,exists
import inception_score, cPickle

def visualize_inception(improved_keras_dir, t_n_epoch, plot=True):
    score = []
    for i in range(t_n_epoch):
        scorefile = join(improved_keras_dir, 'epoch_{}.score'.format(i))
        if not exists(scorefile):
            datafile = join(improved_keras_dir, 'epoch_{}.pkl'.format(i))
            if not exists(datafile):
                break
            with open(datafile) as f:
                sample = cPickle.load(f)
                t_score = get_inception_score_2(sample)[0]
            with open(scorefile, 'w') as f:
                f.write('%f\n' % t_score)
        else:
            with open(scorefile) as f:
                t_score = float(f.readline())
        score.append(t_score)
    if plot:
        plt.plot(range(len(score)), score)
        plt.show()
    else:
        return score

def get_inception_score_2(all_samples):
    #assert(all_samples.shape[1]==3)
    #all_samples = all_samples.transpose(0,2,3,1)
    return inception_score.get_inception_score(list(all_samples), 1)

topdir = sys.argv[1]
num_epoch = sys.argv[2]
score_outputfile = sys.argv[3]
with open(score_outputfile, 'w') as f:
    for x in visualize_inception(topdir, int(num_epoch), plot=False):
        f.write('%f\n' % x)

