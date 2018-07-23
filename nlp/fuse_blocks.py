import tensorflow as tf

from nlp.encode_blocks import MH_Att_encode, CNN_encode
from nlp.nn import masked_reduce_max, masked_reduce_mean, get_var, minus_mask, bilinear_logit


def CNN_fuse(seqs, mask, filter_size, num_units=None, scope=None, reuse=None, **kwargs):
    """
    Simple CNN fusion
    """
    if num_units is None:
        num_units = seqs.get_shape().as_list()[-1]
    with tf.variable_scope(scope or 'CNN_Fuse_Block', reuse=reuse):
        conv_relu = CNN_encode(seqs, filter_size, num_units)
        pooled = Pool_fuse(conv_relu, mask, **kwargs)
    return pooled


def MR_CNN_fuse(seqs, mask, filter_sizes=(3, 4, 5),
                highway=False, scope=None, reuse=None, **kwargs):
    """
    Multi-resolution CNN fusion
    """
    with tf.variable_scope(scope or 'MR_CNN_Block', reuse=reuse):
        outputs = [Pool_fuse(seqs, mask, **kwargs)] if highway else []
        for fz in filter_sizes:
            outputs.append(CNN_fuse(seqs, mask, fz, scope='cnn_fuse_%d' % fz, **kwargs))
        return tf.concat(outputs, axis=1)


def MH_MR_CNN_fuse(seqs, mask, num_heads=3,
                   scope=None, reuse=None, **kwargs):
    """
    Multi-head multi-resolution CNN fusion
    """
    with tf.variable_scope(scope or 'MultiHead_MCNN_Block', reuse=reuse):
        outputs = []
        for j in range(num_heads):
            outputs.append(MR_CNN_fuse(seqs, mask, scope='mr_cnn_block_%d' % j, **kwargs))
        return tf.add_n(outputs)


def MH_MR_CNN_fuse_v2(seqs, mask, num_heads=4, num_units=256,
                      scope=None, reuse=None, **kwargs):
    with tf.variable_scope(scope or 'MultiHead_MCNN_Block', reuse=reuse):
        outputs = []
        part_seqs = tf.split(seqs, num_heads, axis=2)
        for idx, part_seq in enumerate(part_seqs):
            outputs.append(MR_CNN_fuse(part_seq, mask,
                                       num_units=int(num_units / num_heads),
                                       scope='mr_cnn_head_%d' % idx, **kwargs))
        return tf.concat(outputs, axis=1)


def MH_MR_CNN_fuse_v3(seqs, mask, num_heads=4,
                      scope=None, reuse=None, **kwargs):
    with tf.variable_scope(scope or 'MultiHead_MCNN_Block', reuse=reuse):
        outputs = []
        for j in range(num_heads):
            outputs.append(MR_CNN_fuse(seqs, mask, scope='mr_cnn_block_%d' % j, **kwargs))

        return tf.concat(outputs, axis=1)


def MH_MR_CNN_fuse_v4(seqs, mask, num_heads=4,
                      scope=None, reuse=None, **kwargs):
    with tf.variable_scope(scope or 'MultiHead_MCNN_Block', reuse=reuse):
        outputs = []
        for j in range(num_heads):
            outputs.append(MR_CNN_fuse(seqs, mask, scope='mr_cnn_block_%d' % j, **kwargs))

        return bilinear_logit(tf.concat(outputs, axis=1), num_output=1024,
                              first_num_output=2048,
                              first_act_fn=tf.nn.relu)


def Pool_fuse(seqs, mask, reduce='CONCAT_MEAN_MAX', scope=None, reuse=None,
              task_name=None, norm_by_layer=False, dropout_keep=1., **kwargs):
    """
    Simple word embedding fusion
    """
    with tf.variable_scope(scope or 'SWEM_Block', reuse=reuse):
        if reduce == 'MAX':
            pooled = masked_reduce_max(seqs, mask)
        elif reduce == 'MEAN':  # often worst
            pooled = masked_reduce_mean(seqs, mask)
        elif reduce == 'LEVEL_MEAN_MAX':
            # 1. segment sequence in a small window
            # 2. do average pooling on each window
            # 3. do max pooling across all windows
            def reshape_seqs(x, avg_window_size=3, **kwargs):
                B = tf.shape(x)[0]
                L = tf.cast(tf.shape(x)[1], tf.float32)
                D = x.get_shape().as_list()[-1]
                b = tf.transpose(x, [0, 2, 1])
                extra_pads = tf.cast(tf.ceil(L / avg_window_size) * avg_window_size - L, tf.int32)
                c = tf.pad(b, tf.concat([tf.zeros([2, 2], dtype=tf.int32), [[0, extra_pads]]], axis=0))
                return tf.reshape(c, [B, D, avg_window_size, -1])

            # avg pooling with mask, be careful with all zero mask
            d = tf.reduce_sum(reshape_seqs(seqs, **kwargs), axis=2) / \
                tf.reduce_sum(reshape_seqs(tf.expand_dims(mask + 1e-10, axis=-1), **kwargs), axis=2)

            # max pooling
            pooled = tf.reduce_max(d, axis=2)
        elif reduce == 'SELF_ATTENTION':
            D = seqs.get_shape().as_list()[-1]
            query = tf.tile(get_var('%s_query' % (task_name if task_name else ''),
                                    shape=[1, D, 1]), [tf.shape(seqs)[0], 1, 1])
            score = tf.nn.softmax(minus_mask(tf.matmul(seqs, query) / (D ** 0.5), mask), axis=1)  # [B,L,1]
            pooled = tf.reduce_sum(score * seqs, axis=1)  # [B, D]
        elif reduce.startswith('CONCAT_'):
            if reduce == 'CONCAT_MEAN_MAX':  # often best
                pooled = [Pool_fuse(seqs, mask, reduce='MAX', scope='max_pool',
                                    norm_by_layer=norm_by_layer,
                                    dropout_keep=dropout_keep, **kwargs),
                          Pool_fuse(seqs, mask, reduce='MEAN', scope='avg_pool',
                                    norm_by_layer=norm_by_layer,
                                    dropout_keep=dropout_keep, **kwargs)]
            elif reduce == 'CONCAT_ATT_MAX':
                pooled = [Pool_fuse(seqs, mask, reduce='MAX', scope='max_pool',
                                    norm_by_layer=norm_by_layer,
                                    dropout_keep=dropout_keep, **kwargs),
                          Pool_fuse(seqs, mask, reduce='SELF_ATTENTION', scope='att_pool',
                                    norm_by_layer=norm_by_layer,
                                    dropout_keep=dropout_keep, **kwargs)]
            else:
                raise NotImplementedError
            pooled = tf.concat(pooled, axis=1)
            # set them to false as we already did layer norm and dropout
            # in each sub-block
            norm_by_layer = False
            dropout_keep = 1.
        else:
            raise NotImplementedError

        if dropout_keep < 1.:
            pooled = tf.nn.dropout(pooled, keep_prob=dropout_keep)
        if norm_by_layer:
            pooled = tf.contrib.layers.layer_norm(pooled)

        return pooled


def MH_Att_fuse(seqs, mask, scope=None, reuse=None, **kwargs):
    with tf.variable_scope(scope or 'MultiHead_Attention_Fuse_Block', reuse=reuse):
        rep = MH_Att_encode(seqs, seqs, **kwargs)
        return Pool_fuse(rep, mask, **kwargs)


def Stack_MH_Att_fuse(seqs, mask, num_layers=5, num_units=None,
                      concat_output=True,
                      residual=True, scope=None, reuse=None, **kwargs):
    if num_units is None or residual:
        num_units = seqs.get_shape().as_list()[-1]
    pooled_outputs = []
    with tf.variable_scope(scope or 'Stack_MultiHead_Attention_Fuse_Block', reuse=reuse):
        res_rep = None
        iter_rep = seqs
        for idx in range(num_layers):
            with tf.variable_scope("multi_head_pool_%s" % idx):
                iter_rep = MH_Att_encode(iter_rep, iter_rep, num_units, **kwargs)
                # res
                if residual and (res_rep is not None):
                    iter_rep = iter_rep + res_rep
                res_rep = iter_rep
                pooled = Pool_fuse(iter_rep, mask, **kwargs)
                pooled_outputs.append(pooled)
        return tf.concat(pooled_outputs, axis=1) if concat_output else pooled_outputs[-1]


def Stack_CNN_fuse(seqs, mask, filter_sizes=(3, 3, 3), num_units=None,
                   gate=False, residual=True, scope=None, reuse=None,
                   **kwargs):
    if num_units is None or residual:
        num_units = seqs.get_shape().as_list()[-1]
    pooled_outputs = []
    with tf.variable_scope(scope or 'Deep_CNN_Residual_Block', reuse=reuse):
        last_seqs = seqs
        for idx, fz in enumerate(filter_sizes):
            with tf.variable_scope("conv_pool_%s" % idx):
                if gate:
                    # gate
                    next_seqs = CNN_encode(last_seqs, fz, 2 * num_units)
                    map_res_a, map_res_b = tf.split(next_seqs, num_or_size_splits=2, axis=2)
                    next_seqs = map_res_a * tf.nn.sigmoid(map_res_b)
                else:
                    next_seqs = CNN_encode(last_seqs, fz, num_units)
                # res
                if residual:
                    next_seqs += last_seqs
                pooled = Pool_fuse(next_seqs, mask, **kwargs)
                pooled_outputs.append(pooled)
                last_seqs = next_seqs
        return pooled_outputs[-1]


def MH_Stack_CNN_fuse(seqs, mask, num_heads=3, fuse_multihead_combine='ADD',
                      scope=None, reuse=None, **kwargs):
    """
    Multi-head stack CNN fusion
    """
    with tf.variable_scope(scope or 'MultiHead_Stack_CNN_Block', reuse=reuse):
        outputs = []
        for j in range(num_heads):
            outputs.append(Stack_CNN_fuse(seqs, mask, scope='stack_cnn_block_%d' % j, **kwargs))
        if fuse_multihead_combine == 'ADD':
            return tf.add_n(outputs)
        elif fuse_multihead_combine == 'CONCAT':
            return tf.concat(outputs, axis=1)
        else:
            raise NotImplementedError
