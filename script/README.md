## Gaussian experiment
To train a model using SGD-adagrad with lr=5e-3 and regularization=1e-6.

```
cd script
KERAS_BACKEND=theano python wgan_train.py -d ../data/gaussian_m5_v1 -o gaussian_testrun -l 6 -c 4 --optimizer SGD -v 0 --lr 5e-03 -s adagrad -t wgan -p 1e-6 -b 512 --g_interval 5 -e 50 -g --lt 1
```
