# Training GANs with Optimism
Code for reproducing results in [Training GANs with Optimism]()

Hereinafter,  `REPO_HOME` is used to denote the path to this repository.

## Prerequisites
- Keras (2.0.8)
- Theano
- Pillow
- Numpy, Scipy, cPickle, Matplotlib

## DNA generation

#### Unzip the dataset 

```
cd data/
tar -zxvf motif_spikein_ATAGGC_50runs.tar.gz
```

#### Trainng
For each of the optimizatin strategies with each of 3 different learning rates, train 50 WGAN models for 100 epochs. The resulting models are saved under $REPO_HOME/script/DNA.

```
DATADIR=../data/motif_spikein_ATAGGC_50runs
RUN_CMD=''
cd script/

for RUN in {0..49}
do
	for LR in '5e-03' '5e-04' '5e-05'
	do
		
		# SGD
        	KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SGD_lr$LR \
        		--optimizer SGD  -s None --lr $LR
        
		# SGD with Adagrad
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/adagrad_lr$LR \
			--optimizer SGD  -s adagrad --lr $LR

		# SGD with momentum
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/momentum_lr$LR \
			--optimizer SGD  -s None --momentum 0.9 --lr $LR

		# SGD with Nesterov momentum
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/nesterov_lr$LR \
			--optimizer SGD  -s None --momentum 0.9 --nesterov  --lr $LR

		# SOMD (3 different versions)
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv1_lr$LR \
			--optimizer OMDA  -s None -v 1 --lr $LR
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv2_lr$LR \
			--optimizer OMDA  -s None -v 2 --lr $LR
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv3_lr$LR \
			--optimizer OMDA  -s None -v 3 --lr $LR

		# SOMD with 1:1 training ratio (3 different versions)
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv1_ratio1_lr$LR \
			--optimizer OMDA  -s None --g_interval 1 -v 1 --lr $LR
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv2_ratio1_lr$LR \
			--optimizer OMDA  -s None --g_interval 1 -v 2 --lr $LR
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv3_ratio1_lr$LR \
			--optimizer OMDA  -s None --g_interval 1 -v 3 --lr $LR

	done
done
```

#### KL divergence Evaluation
Change the `rundir` argument in this [Notebook](https://github.com/vsyrgkanis/GAN_training/blob/master/notebooks/DNA-eval.ipynb) to $REPO_HOME/script/DNA/ to reproduce the figures.

## CIFAR 10

#### Training

Run the following codes to train a WGAN model for each of the optimization strategies compared in this section. The resulting models are saved under $REPO_HOME/script/cifar10.

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
Change the `rundir` argument in this [Notebook](https://github.com/vsyrgkanis/GAN_training/blob/master/notebooks/CIFAR10-eval.ipynb) to $REPO_HOME/script/cifar10/ to reproduce the figures.

## License
MIT
