from keras import backend as K
from six.moves import zip
from keras.optimizers import Optimizer
from keras.legacy import interfaces

class OFRL(Optimizer):

    def __init__(self, lr=0.01, version=1., decay=0.,
                 schedule=None, m_rho=0.1, adagrad_epsilon=1e-08, **kwargs):
        super(OFRL, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
            self.version = version
            self.schedule = schedule
            self.m_rho = m_rho
            self.adagrad_epsilon = adagrad_epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        shapes = [K.int_shape(p) for p in params]
        predictable = [K.zeros(shape) for shape in shapes]
        grad_sqr_accum = [K.zeros(shape) for shape in shapes]
        scheduler = [K.ones(shape) for shape in shapes]
        self.weights = [self.iterations] + predictable
        for p, g, m, a, s in zip(params, grads, predictable, grad_sqr_accum, scheduler):
            # Update M
            if self.version == 1:
                new_m = g
            elif self.version == 2:
                new_m = (m * self.iteration + g) / (self.interation+1)
            elif self.version == 3:
                new_m = m * self.m_rho + (1-self.m_rho) * g
            else:
                raise ValueError('self.version {} is not recognized'.format(self.version))

            # Update the sum of squared gradient
            new_a = a + K.square(g)

            # Update learning rate schedule
            if self.schedule is None:
                new_s = s
            elif self.schedule == 'adagrad':
                new_s = s / (K.sqrt(new_a) + self.adagrad_epsilon)
                #new_s = s
            else:
                raise ValueError('self.schedule {} is not recognized'.format(self.schedule))

            # Update params
            new_p = self.update_param(p, g, lr, m, new_m, s, new_s)

            # Finally, apply the updates
            self.updates.append(K.update(m, new_m))
            self.updates.append(K.update(a, new_a))
            self.updates.append(K.update(s, new_s))

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def update_param(self, p, g, lr, m, new_m, s, new_s):
        return new_s / s  * p + lr * new_s * (m - new_m - g)

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'decay': float(K.get_value(self.decay)),
                  'schedule': self.schedule,
                  'version': self.version,
                  'm_rho': self.m_rho,
                  'adagrad_epsilon': self.adagrad_epsilon,
                  }
        base_config = super(OFRL, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class OMDA(OFRL):
    def __init__(self, lr=0.01, version=1., decay=0.,
                 schedule=None, m_rho=0.1, adagrad_epsilon=1e-08, **kwargs):
        super(OMDA, self).__init__(**kwargs)

    def update_param(self, p, g, lr, m, new_m, s, new_s):
        return p + lr * (s * m - s * g - new_s * new_m)
