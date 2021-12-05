# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def linear(inputs, output_size, bias, concat=True, dtype=None, scope=None):
    """
    Linear layer
    :param inputs: A Tensor or a list of Tensors with shape [batch, input_size]
    :param output_size: An integer specify the output size
    :param bias: a boolean value indicate whether to use bias term
    :param concat: a boolean value indicate whether to concatenate all inputs
    :param dtype: an instance of tf.DType, the default value is ``tf.float32''
    :param scope: the scope of this layer, the default value is ``linear''
    :returns: a Tensor with shape [batch, output_size]
    :raises RuntimeError: raises ``RuntimeError'' when input sizes do not
                          compatible with each other
    """

    with tf.variable_scope(scope, default_name="linear", values=[inputs]):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        input_size = [item.get_shape()[-1].value for item in inputs]

        if len(inputs) != len(input_size):
            raise RuntimeError("inputs and input_size unmatched!")

        output_shape = tf.concat([tf.shape(inputs[0])[:-1], [output_size]],
                                 axis=0)
        # Flatten to 2D
        inputs = [tf.reshape(inp, [-1, inp.shape[-1].value]) for inp in inputs]

        results = []

        if concat:
            input_size = sum(input_size)
            inputs = tf.concat(inputs, 1)

            shape = [input_size, output_size]
            matrix = tf.get_variable("matrix", shape, dtype=dtype)
            results.append(tf.matmul(inputs, matrix))
        else:
            for i in range(len(input_size)):
                shape = [input_size[i], output_size]
                name = "matrix_%d" % i
                matrix = tf.get_variable(name, shape, dtype=dtype)
                results.append(tf.matmul(inputs[i], matrix))

        output = tf.add_n(results)

        if bias:
            shape = [output_size]
            bias = tf.get_variable("bias", shape, dtype=dtype)
            output = tf.nn.bias_add(output, bias)

        output = tf.reshape(output, output_shape)

        return output


def maxout(inputs, output_size, maxpart=2, use_bias=True, concat=True,
           dtype=None, scope=None):
    """
    Maxout layer
    :param inputs: see the corresponding description of ``linear''
    :param output_size: see the corresponding description of ``linear''
    :param maxpart: an integer, the default value is 2
    :param use_bias: a boolean value indicate whether to use bias term
    :param concat: concat all tensors if inputs is a list of tensors
    :param dtype: an optional instance of tf.Dtype
    :param scope: the scope of this layer, the default value is ``maxout''
    :returns: a Tensor with shape [batch, output_size]
    :raises RuntimeError: see the corresponding description of ``linear''
    """

    candidate = linear(inputs, output_size * maxpart, use_bias, concat,
                       dtype=dtype, scope=scope or "maxout")
    shape = tf.concat([tf.shape(candidate)[:-1], [output_size, maxpart]],
                      axis=0)
    value = tf.reshape(candidate, shape)
    output = tf.reduce_max(value, -1)

    return output


def layer_norm(inputs, epsilon=1e-6, dtype=None, scope=None):
    """
    Layer Normalization
    :param inputs: A Tensor of shape [..., channel_size]
    :param epsilon: A floating number
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string
    :returns: A Tensor with the same shape as inputs
    """
    with tf.variable_scope(scope, default_name="layer_norm", values=[inputs],
                           dtype=dtype):
        channel_size = inputs.get_shape().as_list()[-1]

        scale = tf.get_variable("scale", shape=[channel_size],
                                initializer=tf.ones_initializer())

        offset = tf.get_variable("offset", shape=[channel_size],
                                 initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(inputs, -1, True)
        variance = tf.reduce_mean(tf.square(inputs - mean), -1, True)

        norm_inputs = (inputs - mean) * tf.rsqrt(variance + epsilon)

        return norm_inputs * scale + offset


def smoothed_softmax_cross_entropy_with_logits(**kwargs):
    logits = kwargs.get("logits")
    labels = kwargs.get("labels")
    smoothing = kwargs.get("smoothing") or 0.0
    normalize = kwargs.get("normalize")
    scope = kwargs.get("scope")

    if logits is None or labels is None:
        raise ValueError("Both logits and labels must be provided")

    with tf.name_scope(scope or "smoothed_softmax_cross_entropy_with_logits",
                       values=[logits, labels]):

        labels = tf.reshape(labels, [-1])

        if not smoothing:
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )
            return ce

        # label smoothing
        vocab_size = tf.shape(logits)[1]

        n = tf.to_float(vocab_size - 1)
        p = 1.0 - smoothing
        q = smoothing / n

        soft_targets = tf.one_hot(tf.cast(labels, tf.int32), depth=vocab_size,
                                  on_value=p, off_value=q)
        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                           labels=soft_targets)

        if normalize is False:
            return xentropy

        # Normalizing constant is the best cross-entropy value with soft
        # targets. We subtract it just for readability, makes no difference on
        # learning
        normalizing = -(p * tf.log(p) + n * q * tf.log(q + 1e-20))

        return xentropy - normalizing


# def local_ness(q, k, tmp=1,heads=1, scope=None):
#     def gussi_mask(length,P, D):
#         base_mask = tf.to_float(
#             tf.tile(tf.reshape(tf.range(length), [1, 1, 1, -1]), [batch, heads, length, 1])) - P
#         mask = -tf.pow(base_mask, 2.0) / (2 * tf.pow((D / 2) + 1, 2.0))
#         # mask = mask / tf.reduce_max(mask, -1)
#         return mask
#
#     with tf.variable_scope(scope, default_name='local_ness', values=[q, k]):
#         length = tf.shape(q)[2]
#         batch = tf.shape(q)[0]
#         dim = q.get_shape().as_list()[-1]
#         p_weight = tf.get_variable('p_weight', [8 * dim, 8 * dim])
#         p_U = tf.get_variable('p_U', [8 * dim, heads])
#         P = tf.matmul(tf.tanh(
#             tf.matmul(tf.reshape(tf.transpose(q, [0, 2, 1, 3]), [-1, 8 * dim]), p_weight),
#             'local_P'), p_U)
#         P = tf.transpose(tf.reshape(P, [batch, length, heads,-1]), [0, 2, 1,3]) # [b,h,l,1]
#
#         d_weight = tf.get_variable('d_weight', [8 * dim, 8 * dim])
#         d_U = tf.get_variable('d_U', [8 * dim, heads])
#         D = tf.matmul(tf.tanh(tf.matmul(tf.reshape(tf.transpose(k, [0, 2, 1, 3]), [-1, 8 * dim]), d_weight), 'local_D'), d_U)
#         D = tf.transpose(tf.reshape(D, [batch, length, heads,-1]), [0, 2, 1,3])
#         init_mask = gussi_mask(length, P, D)
#         local_bias = init_mask*1e2
#         return local_bias


def local_ness(q, k, tmp=1, heads=2,num_heads=8, scope=None):
    def gussi_mask(P, D, batch, length):
        base_mask = tf.to_float(
            tf.tile(tf.reshape(tf.range(length), [1, 1, 1, -1]), [batch, heads*num_heads//8, length, 1])) - P
        mask = -tf.pow(base_mask, 2.0) / (2 * tf.pow((D / 2) + 0.5, 2.0))
        # mask = mask / tf.reduce_max(mask, -1)
        return mask

    with tf.variable_scope(scope, default_name='local_ness', values=[q, k]):
        length = tf.shape(q)[2]
        batch = tf.shape(q)[0]
        dim = q.get_shape().as_list()[-1]
        p_weight = tf.get_variable('p_weight', [num_heads * dim, num_heads * dim])
        p_U = tf.get_variable('p_U', [num_heads * dim, heads*num_heads//8])
        P = tf.matmul(tf.tanh(
            tf.matmul(tf.reshape(tf.transpose(q, [0, 2, 1, 3]), [-1, num_heads * dim]), p_weight),
            'local_P'), p_U)
        P = tf.transpose(tf.reshape(P, [batch, length, heads*num_heads//8, -1]), [0, 2, 1, 3])  # [b,h,l,1]

        window_size = tf.get_variable('localness_window_size', [1, heads*num_heads//8, 1, 1], initializer=tf.initializers.ones)
        sigma_w = tf.get_variable('localness_sigma', [num_heads * dim, heads*num_heads//8])

        sigma = tf.tanh(
            tf.transpose(tf.reshape(tf.matmul(tf.reshape(tf.transpose(k, [0, 2, 1, 3]), [-1, num_heads * dim]), sigma_w),
                                    [batch, length, heads*num_heads//8, 1]), [0, 2, 1, 3]))
        D = tf.pow(tf.abs(window_size), sigma)
        init_mask = gussi_mask(P, D, batch, length)
        local_bias = init_mask * 0.1
        return local_bias


# def short_term(inputs, heads=1, scope=None):
#     def gussi_mask(P, D, batch, length):
#         base_mask = tf.to_float(
#             tf.tile(tf.reshape(tf.range(length), [1, 1, 1, -1]), [batch, heads, length, 1])) - P
#         mask = -tf.pow(base_mask, 2.0) / (2 * tf.pow((D/2 +0.5), 2.0))
#         # mask = tf.exp(mask)
#         # mask = mask/tf.reduce_max(mask,-1,keepdims=True)
#         return mask
#
#     with tf.variable_scope(scope, default_name='shorterm'):
#         length = tf.shape(inputs)[2]
#         batch = tf.shape(inputs)[0]
#         dim = inputs.get_shape().as_list()[-1]
#         P = tf.to_float(tf.reshape(tf.range(length), [1,1,-1, 1]))
#         #sigma_U = tf.get_variable('short_sigmaU', [8 * dim, heads],initializer=tf.initializers.random_uniform(minval=0.1))
#         #sigma_W = tf.get_variable('short_sigmaW', [8 * dim, 8 * dim],initializer=tf.initializers.random_uniform(minval=0.1))
#         #sigma = tf.tanh(tf.matmul(tf.reshape(tf.transpose(inputs, [0, 2, 1, 3]), [-1, dim * 8]), sigma_W))
#         #sigma = tf.matmul(sigma, sigma_U) # [b,l,h]
#         #sigma = tf.transpose(tf.reshape(sigma, [batch, length, heads, -1]), [0, 2, 1, 3])
#         sigma_U = tf.get_variable('short_sigma',[1,heads,1,1])
#         sigma = tf.to_float(length)*tf.sigmoid(sigma_U)
#         init_mask = gussi_mask(P, sigma, batch, length)
#         shorterm = init_mask
#         return shorterm

def short_term(inputs, heads=1,num_heads=8, scope=None):
    def gussi_mask(P, D, batch, length):
        base_mask = tf.to_float(
            tf.tile(tf.reshape(tf.range(length), [1, 1, 1, -1]), [batch, heads, length, 1])) - P
        mask = -tf.pow(base_mask, 2.0) / (2 * tf.pow(D / 2 + 0.5, 2.0))
        # mask = mask/tf.reduce_max(mask,-1,keepdims=True)
        return mask

    with tf.variable_scope(scope, default_name='shorterm_v2'):
        length = tf.shape(inputs)[2]
        batch = tf.shape(inputs)[0]
        dim = inputs.get_shape().as_list()[-1]
        P = tf.to_float(tf.reshape(tf.range(length), [1, 1, -1, 1]))
        # get D
        window_size = tf.get_variable('short_window_size', [1, heads*num_heads//8, 1, 1], initializer=tf.initializers.ones)
        sigma_W = tf.get_variable('short_sigma_W', [num_heads * dim, heads*num_heads//8])
        sigma = tf.tanh(tf.transpose(
            tf.reshape(tf.matmul(tf.reshape(tf.transpose(inputs, [0, 2, 1, 3]), [-1, num_heads * dim]), sigma_W),
                       [batch, length, heads*num_heads//8, 1]), [0, 2, 1, 3]))
        # window_size = tf.Print(window_size,[window_size],'window_size:')
        D = tf.pow(tf.abs(window_size), sigma)
        mask = gussi_mask(P, D, batch, length)
        return mask


# def long_term(inputs, heads=1, scope=None):
#     def gussi_mask(P, D, batch, length):
#         base_mask = tf.to_float(
#             tf.tile(tf.reshape(tf.range(length), [1, 1, 1, -1]), [batch, heads, length, 1])) - P
#         mask = -tf.pow(base_mask, 2.0) / (2 * tf.pow((D/2) + 0.1, 2.0))
#         # mask = tf.exp(mask)
#         mini = tf.expand_dims(tf.reduce_min(mask, -1), axis=-1)
#         mask = mini - mask
#         # mask = mask/tf.reduce_max(mask,-1,keepdims=True)
#         return mask
#
#     with tf.variable_scope(scope, default_name='longterm'):
#         length = tf.shape(inputs)[2]
#         batch = tf.shape(inputs)[0]
#         dim = inputs.get_shape().as_list()[-1]
#         P = tf.to_float(tf.reshape(tf.range(length), [1,1,-1, 1]))
#
#         sigma_U = tf.get_variable('long_sigmaU', [8 * dim, heads],
#                                   initializer=tf.initializers.random_uniform(minval=0.1))
#         sigma_W = tf.get_variable('long_sigmaW', [8 * dim, 8 * dim],
#                                   initializer=tf.initializers.random_uniform(minval=0.1))
#         sigma = tf.tanh(tf.matmul(tf.reshape(tf.transpose(inputs, [0, 2, 1, 3]), [-1, dim * 8]), sigma_W))
#         sigma = tf.matmul(sigma, sigma_U)
#         sigma = tf.transpose(tf.reshape(sigma, [batch, length, heads, -1]), [0, 2, 1, 3])
#         init_mask = gussi_mask(P, sigma, batch, length)
#         longterm = init_mask*1e2
#         return longterm


def long_term(inputs, heads=1,num_heads=8, scope=None):
    def gussi_mask(P, D, batch, length):
        base_mask = tf.to_float(
            tf.tile(tf.reshape(tf.range(length), [1, 1, 1, -1]), [batch, heads*num_heads//8, length, 1])) - P
        mask = -tf.pow(base_mask, 2.0) / (2 * tf.pow(D / 2 + 0.5, 2.0))
        mini = tf.expand_dims(tf.reduce_min(mask, -1), axis=-1)
        mask = mini - mask
        # mask = mask/tf.reduce_max(mask,-1,keepdims=True)
        return mask

    with tf.variable_scope(scope, default_name='longterm_v2'):
        length = tf.shape(inputs)[2]
        batch = tf.shape(inputs)[0]
        dim = inputs.get_shape().as_list()[-1]
        P = tf.to_float(tf.reshape(tf.range(length), [1, 1, -1, 1]))
        # get D
        window_size = tf.get_variable('long_window_size', [1, heads*num_heads//8, 1, 1], initializer=tf.initializers.ones)
        sigma_W = tf.get_variable('long_sigma_W', [num_heads * dim, heads*num_heads//8])
        sigma = tf.tanh(tf.transpose(
            tf.reshape(tf.matmul(tf.reshape(tf.transpose(inputs, [0, 2, 1, 3]), [-1, num_heads * dim]), sigma_W),
                       [batch, length, heads*num_heads//8, 1]), [0, 2, 1, 3]))
        # window_size = tf.Print(window_size,[window_size],'window_size:')
        D = tf.pow(tf.abs(window_size), sigma)
        mask = gussi_mask(P, D, batch, length)
        return mask * 1e2


def for_back_ward(inputs, heads=1,num_heads=8, scope=None):
    with tf.variable_scope(scope, default_name='for_back'):
        length = tf.shape(inputs)[2]

        init_mask = tf.ones([length, length])
        forward = tf.tile(tf.expand_dims(tf.matrix_band_part(init_mask, -1, 0), dim=0), [heads*num_heads//8, 1, 1])
        backward = 1 - forward

        return tf.tile(tf.expand_dims(tf.concat([forward, backward], axis=0), axis=0), [tf.shape(inputs)[0], 1, 1, 1])


def temperature(q, k, lambda_t=4, heads=1, scope=None):
    with tf.variable_scope(scope, default_name='temperature'):
        # init lambda
        length = tf.shape(q)[2]
        batch = tf.shape(q)[0]
        dim = q.get_shape().as_list()[-1]
        lambda_t = tf.ones([batch, heads, length, length]) * lambda_t
        weigth_q = tf.get_variable('weigth_q', [8 * dim, heads * dim], initializer=tf.random_normal_initializer)
        weigth_k = tf.get_variable('weigth_k', [8 * dim, heads * dim], initializer=tf.random_normal_initializer)
        q_tmp = tf.transpose(tf.reshape(tf.matmul(tf.reshape(tf.transpose(q, [0, 2, 1, 3]), [-1, 8 * dim]), weigth_q),
                                        [batch, length, heads, -1]), [0, 2, 1, 3])
        k_tmp = tf.transpose(tf.reshape(tf.matmul(tf.reshape(tf.transpose(k, [0, 2, 1, 3]), [-1, 8 * dim]), weigth_k),
                                        [batch, length, heads, -1]), [0, 2, 1, 3])
        beta = tf.tanh(tf.matmul(q_tmp, k_tmp, transpose_b=True))
        temper = tf.pow(lambda_t, beta)
        return temper


def gate_sum(x, y, q, heads=8, is_bid=False, type='none', scope='None'):
    with tf.variable_scope(scope, default_name="gate_sum", values=[x, y]):
        shape = x.get_shape().as_list()[-1]
        gate_w = tf.get_variable('gate_weight', shape=[8*shape, 8*shape],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        gate_U = tf.get_variable('gate_u', shape=[8*shape, heads])
        # gate_b = tf.get_variable('gate_bias', shape=[shape],
        #                          initializer=tf.initializers.zeros)
        gate = tf.transpose(tf.sigmoid(
            tf.reshape(
                tf.matmul(tf.matmul(tf.reshape(tf.transpose(q, [0, 2, 1, 3]), [-1, 8*shape]), gate_w), gate_U),
                [tf.shape(x)[0], tf.shape(x)[2], heads, 1])),[0,2,1,3])
        out = x * gate + y * (1 - gate)
        if is_bid:
            biaffi_w = tf.get_variable('bi_werght', shape=[shape, shape],
                                       initializer=tf.initializers.uniform_unit_scaling)
            out = tf.matmul(tf.reshape(out, [-1, shape]), biaffi_w)
        if type == 'tanh':
            out = tf.tanh(out)

        return tf.reshape(out, tf.shape(x))

def sub_gate_sum2(x, y, q, heads=8, is_bid=False, type='none', scope=None):
    with tf.variable_scope(scope, default_name="sub_gate_sum", values=[x, y]):
        shape = x.get_shape().as_list()[-1]
        gate_w = tf.get_variable('gate_weight', shape=[8*shape, 8*shape],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        gate_U = tf.get_variable('gate_u', shape=[8*shape, heads])
        # gate_b = tf.get_variable('gate_bias', shape=[shape],
        #                          initializer=tf.initializers.zeros)
        gate = tf.transpose(tf.sigmoid(
            tf.reshape(
                tf.matmul(tf.tanh(tf.matmul(tf.reshape(tf.transpose(q, [0, 2, 1, 3]), [-1, 8*shape]), gate_w)), gate_U),
                [tf.shape(x)[0], tf.shape(x)[2], heads, 1])),[0,2,1,3])
        out = x[:,:heads,:,:] * gate + y[:,:heads,:,:] * (1 - gate)
        # out = tf.concat([out,y[:,5:,:,:]],axis=1)
        if is_bid:
            biaffi_w = tf.get_variable('bi_werght', shape=[shape, shape],
                                       initializer=tf.initializers.uniform_unit_scaling)
            out = tf.matmul(tf.reshape(out, [-1, shape]), biaffi_w)
        if type == 'tanh':
            out = tf.tanh(out)

        return tf.reshape(out, tf.shape(x))

def sub_gate_sum(x, y, q, heads=8, num_heads=8,is_bid=False, type='none', scope=None):
    with tf.variable_scope(scope, default_name="sub_gate_sum", values=[x, y]):
        shape = x.get_shape().as_list()[-1]
        # gate_w = tf.get_variable('gate_weight', shape=[8*shape, 8*shape],
        #                          initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        gate_U = tf.get_variable('gate_u', shape=[num_heads*shape, num_heads])
        # gate_b = tf.get_variable('gate_bias', shape=[shape],
        #                          initializer=tf.initializers.zeros)
        dimss = q.get_shape().as_list()
        if len(dimss)==4:
            gate = tf.transpose(tf.sigmoid(
                tf.reshape(
                    tf.matmul(tf.reshape(tf.transpose(q, [0, 2, 1, 3]), [-1, num_heads*shape]), gate_U),
                    [tf.shape(x)[0], tf.shape(x)[2], num_heads, 1])),[0,2,1,3])
        else:
            gate = tf.transpose(tf.sigmoid(
                tf.reshape(
                    tf.matmul(tf.reshape(q, [-1, num_heads * shape]),gate_U),
                    [tf.shape(x)[0], tf.shape(x)[2], num_heads, 1])), [0, 2, 1, 3])
        out = x[:,:num_heads,:,:] * gate + y[:,:num_heads,:,:] * (1 - gate)
        # out = tf.concat([out,y[:,5:,:,:]],axis=1)
        if is_bid:
            biaffi_w = tf.get_variable('bi_werght', shape=[shape, shape],
                                       initializer=tf.initializers.uniform_unit_scaling)
            out = tf.matmul(tf.reshape(out, [-1, shape]), biaffi_w)
        if type == 'tanh':
            out = tf.tanh(out)

        return tf.reshape(out, tf.shape(x))

def sub_gate_sum_m(x, y, q, heads=8, is_bid=False, type='none', scope=None):
    with tf.variable_scope(scope, default_name="sub_gate_sum_merge", values=[x, y]):
        shape = x.get_shape().as_list()[-1]
        dims = q.get_shape().as_list()[-1]
        gate_w = tf.get_variable('gate_weight', shape=[8*dims, dims],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        gate_U = tf.get_variable('gate_u', shape=[dims, heads],initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        # gate_b = tf.get_variable('gate_bias', shape=[shape],
        #                          initializer=tf.initializers.zeros)
        gate = tf.transpose(tf.sigmoid(
            tf.reshape(
                tf.matmul(tf.tanh(tf.matmul(tf.reshape(tf.transpose(q, [0, 2, 1, 3]), [-1, 8*dims]),gate_w)),gate_U),
                [tf.shape(x)[0], tf.shape(x)[2], heads, 1])),[0,2,1,3])
        out = x[:,:8,:,:] * gate + y[:,:8,:,:] * (1 - gate)
        #out = tf.concat([out,y[:,5:,:,:]],axis=1)
        if is_bid:
            biaffi_w = tf.get_variable('bi_werght', shape=[shape, shape],
                                       initializer=tf.initializers.uniform_unit_scaling)
            out = tf.matmul(tf.reshape(out, [-1, shape]), biaffi_w)
        if type == 'tanh':
            out = tf.tanh(out)

        return tf.reshape(out, tf.shape(x))
