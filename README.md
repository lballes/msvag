# (Momentum) Stochastic Variance-Adapted Gradient, (M-)SVAG

This is a [TensorFlow](https://www.tensorflow.org/) implementation of the M-SVAG and SVAG optimization algorithms described in the paper [Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients][1]

## Installation

Install via

    pip install git+https://github.com/lballes/msvag.git

``msvag`` requires a TensorFlow installation (the code has been tested for versions 1.4 and higher), but this is *not* currently enforced in the ``setup.py`` to allow for either the CPU or the GPU version.

## Usage

The ``msvag`` module contains the two classes ``MSVAGOptimizer`` and ``SVAGOptimizer``, which inherit from ``tf.train.Optimizer`` and can be used as direct drop-in replacement for TensorFlow's built-in optimizers.

    from msvag import MSVAGOptimizer
    
    loss = ...
    opt = MSVAGOptimizer(learning_rate=0.1, beta=0.9)
    step = opt.minimize(loss)
    with tf.Session() as sess:
        sess.run([loss, step])

SVAG and M-SVAG have two hyper-parameters: a learning rate (``learning_rate``) and a moving average constant (``beta``). The default value ``beta=0.9`` should work for most problems.

## Short Description of (M-)SVAG

We give a short description of the two algorithms, ignoring various details. Please refer to the [paper][1] for a complete description.

M-SVAG and SVAG maintain exponential moving averages of past stochastic gradients and their element-wise square

    m = beta*m + (1-beta)*g
    v = beta*v + (1-beta)*g**2

and obtain an estimate of the stochastic gradient variance via

    s = (v-m**2)/(1-rho)

where ``rho`` is a scalar factor (see paper). We then compute variance adaptation factors

    gamma = m**2/(m**2 + rho*s)              # for M-SVAG
    gamma = m**2/(m**2 + s)                  # for SVAG

and update

    theta = theta - learning_rate*gamma*m    # for M-SVAG
    theta = theta - learning_rate*gamma*g    # for SVAG

## Feedback

If you have any questions or suggestions regarding this implementation, please open an issue. Apart from that, we welcome any feedback regarding the performance of (M-)SVAG on your training problems (mail to lballes@tue.mpg.de).

## Citation

If you use (M-)SVAG for your research, please cite the [paper][1].


[1]: TODO


