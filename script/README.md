# TRAINING GANS WITH OPTIMISM
Code for reproducing results in [TRAINING GANS WITH OPTIMISM]()

Hereinafter,  `REPO_HOME` is used to denote the path to this repository.

## DNA generation

#### Unzip the dataset 

```
cd data/
tar -zxvf motif_spikein_ATAGGC_50runs.tar.gz
```

#### Trainng
For each of the optimizatin strategies with each of 3 different learning rates, train 50 WGAN models for 100 epochs. The resulting models are saved under $REPO_HOME/DNA.

```
DATADIR=../data/motif_spikein_ATAGGC_50runs
cd script/

for RUN in {0..49}
do
	for LR in '5e-03' '5e-04' '5e-05'
	do
		# SGD
        python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SGD_lr$LR --optimizer SGD  -s None --lr $LR
        
        # SGD with Adagrad
        python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/adagrad_lr$LR --optimizer SGD  -s adagra --lr $LR
        
        # SGD with momentum
        python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/momentum_lr$LR --optimizer SGD  -s None --momentum 0.9 --lr $LR
        
        # SGD with Nesterov momentum
        python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/nesterov_lr$LR --optimizer SGD  -s None --momentum 0.9 --nestero  --lr $LR
        
        # SOMD (3 different versions)
        python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv1_lr$LR --optimizer OMDA  -s None -v 1 --lr $LR
        python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv2_lr$LR --optimizer OMDA  -s None -v 2 --lr $LR
        python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv3_lr$LR --optimizer OMDA  -s None -v 3 --lr $LR
        
        # SOMD with 1:1 training ratio (3 different versions)
        python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv1_ratio1_lr$LR --optimizer OMDA  -s None --g_interval 1 -v 1 --lr $LR
        python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv2_ratio1_lr$LR --optimizer OMDA  -s None --g_interval 1 -v 2 --lr $LR
        python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv3_ratio1_lr$LR --optimizer OMDA  -s None --g_interval 1 -v 3 --lr $LR
	done
done
```

#### KL divergence Evaluation
Change the `rundir` argument in this [Notebook](https://github.com/vsyrgkanis/GAN_training/blob/master/notebooks/DNA.ipynb) to $REPO_HOME/script/DNA/ to reproduce the figures.

## CIFAR 10

#### Training

Run the following codes to train a WGAN model for each of the optimization strategies compared in this section. The resulting models are saved under $REPO_HOME/cifar10.

```
cd script/

## Adam
python cifar10.py -o cifar10/adam --optimizer SGD -v 1 --schedule adam

## Adam with ratio=1
python cifar10.py -o cifar10/adam_ratio1 --optimizer SGD -v 1 --schedule adam --training_ratio  1

## optimAdam
python cifar10.py -o cifar10/optimAdam --optimizer optimAdam

## optimAdam with ratio=1
python cifar10.py -o cifar10/optimAdam_ratio1 --optimizer optimAdam  --training_ratio 1
```

#### Evaluation
Change the `rundir` argument in this [Notebook](https://github.com/vsyrgkanis/GAN_training/blob/master/notebooks/cifar10.ipynb) to $REPO_HOME/script/cifar10/ to reproduce the figures.