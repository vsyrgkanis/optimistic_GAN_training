from keras import backend as K
from six.moves import zip
from keras.optimizers import Optimizer


class SGD_v1(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):
        super(SGD_v1, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        #self.beta_1 = K.variable(beta_1, name='beta_1')
        #self.beta_2 = K.variable(beta_2, name='beta_2')
        #self.epsilon = epsilon
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        #t = self.iterations + 1
        #lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
        #             (1. - K.pow(self.beta_1, t)))

        shapes = [K.get_variable_shape(p) for p in params]
        grads_m1 = [K.zeros(shape) for shape in shapes]
        #ms = [K.zeros(shape) for shape in shapes]
        #vs = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + grads_m1

        for p, g, gm1 in zip(params, grads, grads_m1):
            #m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            #v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            gall = 2 * g - gm1
            p_t = p - lr * gall
            #p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(gm1, g))
            #self.updates.append(K.update(m, m_t))
            #self.updates.append(K.update(v, v_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  #'beta_1': float(K.get_value(self.beta_1)),
                  #'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay))}
                  #'epsilon': self.epsilon}
        base_config = super(SGD_v1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RMSprop_v1(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, rho=0.9, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):
        super(RMSprop_v1, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        #self.beta_1 = K.variable(beta_1, name='beta_1')
        #self.beta_2 = K.variable(beta_2, name='beta_2')
        self.rho = K.variable(rho, name='rho')
        self.epsilon = epsilon
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        #t = self.iterations + 1
        #lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
        #             (1. - K.pow(self.beta_1, t)))

        shapes = [K.get_variable_shape(p) for p in params]
        grads_m1 = [K.zeros(shape) for shape in shapes]
        #ms = [K.zeros(shape) for shape in shapes]
        #vs = [K.zeros(shape) for shape in shapes]
        accumulators = [K.zeros(shape) for shape in shapes]
        lr_m1 = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + grads_m1 + accumulators + lr_m1

        for p, g, gm1, a, lm1 in zip(params, grads, grads_m1, accumulators, lr_m1):
            #m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            #v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            lr_t = lr / (K.sqrt(new_a) + self.epsilon)

            p_t = p - 2 * lr_t * g  + lm1 * gm1
            #p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(gm1, g))
            self.updates.append(K.update(a, new_a))
            self.updates.append(K.update(lm1, lr_t))
            #self.updates.append(K.update(m, m_t))
            #self.updates.append(K.update(v, v_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  #'beta_1': float(K.get_value(self.beta_1)),
                  #'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'rho': float(K.get_value(self.rho)),
                  'epsilon': self.epsilon}
        base_config = super(RMSprop_v1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RMSprop_v1_1(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, rho=0.9, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):
        super(RMSprop_v1_1, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        #self.beta_1 = K.variable(beta_1, name='beta_1')
        #self.beta_2 = K.variable(beta_2, name='beta_2')
        self.rho = K.variable(rho, name='rho')
        self.epsilon = epsilon
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        #t = self.iterations + 1
        #lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
        #             (1. - K.pow(self.beta_1, t)))

        shapes = [K.get_variable_shape(p) for p in params]
        grads_m1 = [K.zeros(shape) for shape in shapes]
        #ms = [K.zeros(shape) for shape in shapes]
        #vs = [K.zeros(shape) for shape in shapes]
        accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + grads_m1 + accumulators

        for p, g, gm1, a in zip(params, grads, grads_m1, accumulators):
            #m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            #v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            g_t = 2 * g - gm1
            new_a = self.rho * a + (1. - self.rho) * K.square(g_t )
            lr_t = lr / (K.sqrt(new_a) + self.epsilon)

            p_t = p - lr_t * g_t
            #p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(gm1, g))
            self.updates.append(K.update(a, new_a))
            #self.updates.append(K.update(m, m_t))
            #self.updates.append(K.update(v, v_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  #'beta_1': float(K.get_value(self.beta_1)),
                  #'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'rho': float(K.get_value(self.rho)),
                  'epsilon': self.epsilon}
        base_config = super(RMSprop_v1_1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RMSprop_v1_2(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, rho=0.9, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):
        super(RMSprop_v1_2, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        #self.beta_1 = K.variable(beta_1, name='beta_1')
        #self.beta_2 = K.variable(beta_2, name='beta_2')
        self.rho = K.variable(rho, name='rho')
        self.epsilon = epsilon
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        #t = self.iterations + 1
        #lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
        #             (1. - K.pow(self.beta_1, t)))

        shapes = [K.get_variable_shape(p) for p in params]
        grads_m1 = [K.zeros(shape) for shape in shapes]
        lr_m1 = [K.ones(shape) for shape in shapes]
        #ms = [K.zeros(shape) for shape in shapes]
        #vs = [K.zeros(shape) for shape in shapes]
        accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + grads_m1 + accumulators

        for p, g, gm1, a, lm1 in zip(params, grads, grads_m1, accumulators, lr_m1):
            #m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            #v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            lr_t = lr / (K.sqrt(new_a) + self.epsilon)
            p_t = p * lr_t / lm1 - 2 * lr_t * g + lm1 * gm1
            #p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(gm1, g))
            self.updates.append(K.update(a, new_a))
            self.updates.append(K.update(lm1, lr_t))
            #self.updates.append(K.update(m, m_t))
            #self.updates.append(K.update(v, v_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  #'beta_1': float(K.get_value(self.beta_1)),
                  #'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'rho': float(K.get_value(self.rho)),
                  'epsilon': self.epsilon}
        base_config = super(RMSprop_v1_2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SGD_v2(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):
        super(SGD_v2, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        #self.beta_1 = K.variable(beta_1, name='beta_1')
        #self.beta_2 = K.variable(beta_2, name='beta_2')
        #self.epsilon = epsilon
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        #t = self.iterations + 1
        #lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
        #             (1. - K.pow(self.beta_1, t)))

        t = self.iterations + 2.0
        shapes = [K.get_variable_shape(p) for p in params]
        #ms = [K.zeros(shape) for shape in shapes]
        #vs = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations]

        for p, g in zip(params, grads):
            #m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            #v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = (t*t - 1)*p/(t*t) - (t+1) * lr * g / t
            #p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            #self.updates.append(K.update(m, m_t))
            #self.updates.append(K.update(v, v_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  #'beta_1': float(K.get_value(self.beta_1)),
                  #'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay))}
                  #'epsilon': self.epsilon}
        base_config = super(SGD_v2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SGD_v3(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):
        super(SGD_v3, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        #self.beta_1 = K.variable(beta_1, name='beta_1')
        #self.beta_2 = K.variable(beta_2, name='beta_2')
        #self.epsilon = epsilon
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        #t = self.iterations + 1
        #lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
        #             (1. - K.pow(self.beta_1, t)))

        t = self.iterations + 2.0
        shapes = [K.get_variable_shape(p) for p in params]
        accum = [K.zeros(shape) for shape in shapes]
        #ms = [K.zeros(shape) for shape in shapes]
        #vs = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations]

        for p, g, acc in zip(params, grads, accum):
            #m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            #v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            acc_t = acc + g
            p_t = (t - 1)*p/(t+1) - 2/(t+1) * lr * acc_t - lr * g
            #p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(acc, acc_t))
            #self.updates.append(K.update(m, m_t))
            #self.updates.append(K.update(v, v_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  #'beta_1': float(K.get_value(self.beta_1)),
                  #'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay))}
                  #'epsilon': self.epsilon}
        base_config = super(SGD_v3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SGD_v4(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., rho=0.9, **kwargs):
        super(SGD_v4, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        #self.beta_1 = K.variable(beta_1, name='beta_1')
        #self.beta_2 = K.variable(beta_2, name='beta_2')
        #self.epsilon = epsilon
        self.rho = rho
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        #t = self.iterations + 1
        #lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
        #             (1. - K.pow(self.beta_1, t)))

        shapes = [K.get_variable_shape(p) for p in params]
        accum = [K.zeros(shape) for shape in shapes]
        #ms = [K.zeros(shape) for shape in shapes]
        #vs = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations]

        for p, g, acc in zip(params, grads, accum):
            #m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            #v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            acc_t = acc*self.rho + (1-self.rho)*g
            p_t = p - lr * (acc_t-acc) - lr * g
            #p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(acc, acc_t))
            #self.updates.append(K.update(m, m_t))
            #self.updates.append(K.update(v, v_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  #'beta_1': float(K.get_value(self.beta_1)),
                  #'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay))}
                  #'epsilon': self.epsilon}
        base_config = super(SGD_v4, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RMSprop_v2(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, rho=0.9, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):
        super(RMSprop_v2, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        #self.beta_1 = K.variable(beta_1, name='beta_1')
        #self.beta_2 = K.variable(beta_2, name='beta_2')
        self.rho = K.variable(rho, name='rho')
        self.epsilon = epsilon
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 2.0
        #lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
        #             (1. - K.pow(self.beta_1, t)))

        shapes = [K.get_variable_shape(p) for p in params]
        #ms = [K.zeros(shape) for shape in shapes]
        #vs = [K.zeros(shape) for shape in shapes]
        accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations]  + accumulators

        for p, g, a in zip(params, grads, accumulators):
            #m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            #v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            lr_t = lr / (K.sqrt(new_a) + self.epsilon)

            p_t = (t*t-1) * p / (t*t) - lr_t * g * (t+1) / t
            #p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(a, new_a))
            #self.updates.append(K.update(m, m_t))
            #self.updates.append(K.update(v, v_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  #'beta_1': float(K.get_value(self.beta_1)),
                  #'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'rho': float(K.get_value(self.rho)),
                  'epsilon': self.epsilon}
        base_config = super(RMSprop_v2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RMSprop_v3(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, rho=0.9, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):
        super(RMSprop_v3, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        #self.beta_1 = K.variable(beta_1, name='beta_1')
        #self.beta_2 = K.variable(beta_2, name='beta_2')
        self.rho = K.variable(rho, name='rho')
        self.epsilon = epsilon
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 2.0
        #lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
        #             (1. - K.pow(self.beta_1, t)))

        shapes = [K.get_variable_shape(p) for p in params]
        #ms = [K.zeros(shape) for shape in shapes]
        #vs = [K.zeros(shape) for shape in shapes]
        accumulators = [K.zeros(shape) for shape in shapes]
        accumulators_g = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + accumulators +  accumulators_g

        for p, g, a, ag in zip(params, grads, accumulators, accumulators_g):
            #m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            #v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            lr_t = lr / (K.sqrt(new_a) + self.epsilon)

            ag_t = ag + g

            p_t = (t-1) * p / (t+1) - lr_t * 2 / (t+1) * ag_t - lr_t *g
            #p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(a, new_a))
            self.updates.append(K.update(ag, ag_t))
            #self.updates.append(K.update(m, m_t))
            #self.updates.append(K.update(v, v_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  #'beta_1': float(K.get_value(self.beta_1)),
                  #'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'rho': float(K.get_value(self.rho)),
                  'epsilon': self.epsilon}
        base_config = super(RMSprop_v3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RMSprop_v4(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, rho=0.9, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., rho_g = 0.9, **kwargs):
        super(RMSprop_v4, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        #self.beta_1 = K.variable(beta_1, name='beta_1')
        #self.beta_2 = K.variable(beta_2, name='beta_2')
        self.rho = K.variable(rho, name='rho')
        self.rho_g = K.variable(rho_g, name='rho_g')
        self.epsilon = epsilon
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 2.0
        #lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
        #             (1. - K.pow(self.beta_1, t)))

        shapes = [K.get_variable_shape(p) for p in params]
        #ms = [K.zeros(shape) for shape in shapes]
        #vs = [K.zeros(shape) for shape in shapes]
        accumulators = [K.zeros(shape) for shape in shapes]
        accumulators_g = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + accumulators + accumulators_g

        for p, g, a, ag in zip(params, grads, accumulators, accumulators_g):
            #m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            #v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            lr_t = lr / (K.sqrt(new_a) + self.epsilon)

            ag_t = ag*self.rho_g + (1-self.rho_g)*g

            p_t = p - lr_t * g - lr_t * (ag_t - ag)
            #p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(a, new_a))
            self.updates.append(K.update(ag, ag_t))
            #self.updates.append(K.update(m, m_t))
            #self.updates.append(K.update(v, v_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  #'beta_1': float(K.get_value(self.beta_1)),
                  #'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'rho': float(K.get_value(self.rho)),
                  'epsilon': self.epsilon}
        base_config = super(RMSprop_v4, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
