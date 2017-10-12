from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tf_client import base_runner
from merged_sparse_embedding.operations import merged_embedding_lookup_sparse
from merged_sparse_embedding.operations import step_merged_embedding_lookup_sparse
import base_summary


def inference(tensors, stepsize, keep_prob, ps_num, custom_getter):
    # ============     Common Config Setting     =================
    # nfm related settings
    dense_ini_val = 2
    dense_stddev = 0.36
    embedding_ini_val = 1
    embedding_stddev = 0.0002

    # ==============   NET INPUT DATA PREPARE  ================
    # Load feature data
    label = tensors.data[0]  # tensor: [batchsize, 1 x window_size]
    dense = tensors.data[1]  # tensor:[batchsize, dense_dim x window_size]
    query = tensors.data[2:12]  # python-list: [[tensor1, tensor2, ..., tensor-window-size]1, [window_size]2, ... [window_size]10], tensor-shape:[batch-size, sparse-dim]
    user = tensors.data[12:22]
    ad = tensors.data[22:29]
    query_ad = tensors.data[29:34]
    user_ad = tensors.data[34:36]
    context = tensors.data[36:37]
    mask_dense = tensors.data[37]  # [batchsize, maskdim x window_size]
    realsize = tensors.data[len(tensors.data) - 1]  # shape: [batchsize , 1]

    # Load dim data
    label_dim = tensors.dim[0]  # int: 1
    dense_dim = tensors.dim[1]  # int: 431
    query_dim = tensors.dim[2:12]  # python-list: [dim1, dim2,..., dim10]
    user_dim = tensors.dim[12:22]
    ad_dim = tensors.dim[22:29]
    query_ad_dim = tensors.dim[29:34]
    user_ad_dim = tensors.dim[34:36]
    context_dim = tensors.dim[36:37]
    mask_dim = tensors.dim[37]  # int: 34

    # Transform sparse data into form of "[sample1's data, sample2's data, sample3'data, ....]"
    q_field_len = len(query)
    u_field_len = len(user)
    a_field_len = len(ad)
    qa_field_len = len(query_ad)
    ua_field_len = len(user_ad)
    c_field_len = len(context)

    single_sample_sparse_data_len = q_field_len + u_field_len + a_field_len + qa_field_len + ua_field_len + c_field_len
    sample_sparse_dim_len = len(query_dim) + len(user_dim) + len(ad_dim) + len(query_ad_dim) + ua_field_len + c_field_len
    sample_sparse_mask_shape = []

    dim = 16
    total_embedding_dim = 0
    multi_sample_lookup_embeddings = []

    sparse_dims_list = [query_dim, user_dim, ad_dim, query_ad_dim, user_ad_dim, context_dim]
    sparse_dims = []
    for i in range(len(sparse_dims_list)):
        sparse_dims.extend(sparse_dims_list[i])
    sparse_datas = []
    for i in range(stepsize):
        one_sample_sparse_data = []
        sample_sparse_mask_shape.append(mask_dim)
        for q, d in zip(query, query_dim):
            one_sample_sparse_data.append(q[i])
        for u, d in zip(user, user_dim):
            one_sample_sparse_data.append(u[i])
        for a, d in zip(ad, ad_dim):
            one_sample_sparse_data.append(a[i])
        for qa, d in zip(query_ad, query_ad_dim):
            one_sample_sparse_data.append(qa[i])
        for ua, d in zip(user_ad, user_ad_dim):
            one_sample_sparse_data.append(ua[i])
        for c, d in zip(context, context_dim):
            one_sample_sparse_data.append(c[i])
        sparse_datas.extend(one_sample_sparse_data)
        # LookUp Embeddinga
    sparse_datas.append(mask_dense)
    '''
    with tf.variable_scope("embedding_sparse"):
        multi_sample_sparse_embedding = step_merged_sparse_embedding(
                    sparse_datas,
                    sparse_dims,
                    dim,
                    ps_num,
                    init_val=embedding_ini_val,
                    stddev=embedding_stddev)
        for i in range(stepsize):
            multi_sample_lookup_embeddings.extend(multi_sample_sparse_embedding[i])

    sample_mask_tensors = tf.split(mask_dense, sample_sparse_mask_shape, 1)
    # Nfm Embdeddings
    rnn_embedding = []
    for i in range(stepsize):
        single_sample = multi_sample_lookup_embeddings[
                        i * single_sample_sparse_data_len: (i + 1) * single_sample_sparse_data_len]

        q_embedding_list = []
        u_embedding_list = []
        ad_embedding_list = []
        qa_embedding_list = []
        ua_embedding_list = []
        c_embedding_list = []

        q_embedding_list.extend(single_sample[0: q_field_len])
        u_embedding_list.extend(single_sample[q_field_len: (q_field_len + u_field_len)])
        ad_embedding_list.extend(single_sample[(q_field_len + u_field_len): (q_field_len + u_field_len + a_field_len)])
        qa_embedding_list.extend(single_sample[(q_field_len + u_field_len + a_field_len): (
            q_field_len + u_field_len + a_field_len + qa_field_len)])
        ua_embedding_list.extend(single_sample[(q_field_len + u_field_len + a_field_len + qa_field_len):(
            q_field_len + u_field_len + a_field_len + qa_field_len + ua_field_len)])
        c_embedding_list.extend(single_sample[(q_field_len + u_field_len + a_field_len + qa_field_len + ua_field_len):(
            q_field_len + u_field_len + a_field_len + qa_field_len + ua_field_len + c_field_len
        )])

        all_embedding_list = []
        all_embedding_list.extend(q_embedding_list)
        all_embedding_list.extend(u_embedding_list)
        all_embedding_list.extend(ad_embedding_list)
        all_embedding_list.extend(qa_embedding_list)
        all_embedding_list.extend(ua_embedding_list)
        all_embedding_list.extend(c_embedding_list)

        single_sample_mask = sample_mask_tensors[i]

        embedding_scheme_num = q_field_len + u_field_len + a_field_len + qa_field_len + ua_field_len + c_field_len
        all_embedding_list_reweight = [tf.multiply(all_embedding_list[k], tf.reshape(single_sample_mask[:, k], [-1, 1]))
                                       for k in range(embedding_scheme_num)]
        q_embedding_list = all_embedding_list_reweight[0:q_field_len]
        total_qu_len = q_field_len + u_field_len
        u_embedding_list = all_embedding_list_reweight[q_field_len:total_qu_len]
        total_qua_len = total_qu_len + a_field_len
        ad_embedding_list = all_embedding_list_reweight[total_qu_len:total_qua_len]
        qa_embedding_list = all_embedding_list_reweight[total_qua_len:total_qua_len + qa_field_len]
        ua_embedding_list = all_embedding_list_reweight[total_qua_len + qa_field_len:total_qua_len + qa_field_len + ua_field_len]
        c_embedding_list = all_embedding_list_reweight[total_qua_len + qa_field_len + ua_field_len:total_qua_len + qa_field_len + ua_field_len + c_field_len]

        with tf.variable_scope("bi_interaction_afm"):
            qu_embedding_list = []
            qu_embedding_list.append(q_embedding_list)
            qu_embedding_list.append(u_embedding_list)
            bi_qu, bi_quw = bi_interaction_fmwffm(attr_fs=qu_embedding_list,
                                                  ps_num=1,
                                                  init_val=dense_ini_val,
                                                  biscopename="bi_qu",
                                                  sampleIdx=i)
            base_summary.f_summary_add_to_collection(bi_qu)
            base_summary.f_summary_add_to_collection_gradient(bi_qu)
            base_summary.f_summary_add_to_collection(bi_quw)
            base_summary.f_summary_add_to_collection_gradient(bi_quw)

            qad_embedding_list = []
            qad_embedding_list.append(q_embedding_list)
            qad_embedding_list.append(ad_embedding_list)
            bi_qad, bi_qadw = bi_interaction_fmwffm(attr_fs=qad_embedding_list,
                                                    ps_num=1,
                                                    init_val=dense_ini_val,
                                                    biscopename="bi_qad",
                                                    sampleIdx=i)
            base_summary.f_summary_add_to_collection(bi_qad)
            base_summary.f_summary_add_to_collection_gradient(bi_qad)
            base_summary.f_summary_add_to_collection(bi_qadw)
            base_summary.f_summary_add_to_collection_gradient(bi_qadw)

            uad_embedding_list = []
            uad_embedding_list.append(u_embedding_list)
            uad_embedding_list.append(ad_embedding_list)
            bi_uad, bi_uadw = bi_interaction_fmwffm(attr_fs=uad_embedding_list,
                                                    ps_num=1,
                                                    init_val=dense_ini_val,
                                                    biscopename="bi_uad",
                                                    sampleIdx=i)
            base_summary.f_summary_add_to_collection(bi_uad)
            base_summary.f_summary_add_to_collection_gradient(bi_uad)
            base_summary.f_summary_add_to_collection(bi_uadw)
            base_summary.f_summary_add_to_collection_gradient(bi_uadw)

        all_embedding_list2 = [bi_qu, bi_qad, bi_uad]
        all_embedding_list3 = [bi_quw, bi_qadw, bi_uadw]

        all_embedding_list1 = []
        all_embedding_list1.extend(q_embedding_list)
        all_embedding_list1.extend(u_embedding_list)
        all_embedding_list1.extend(ad_embedding_list)
        all_embedding_list1.extend(qa_embedding_list)
        all_embedding_list1.extend(ua_embedding_list)
        all_embedding_list1.extend(c_embedding_list)
        total_embedding_dim = (len(all_embedding_list1) + len(all_embedding_list2) + len(all_embedding_list3)) * dim

        all_embedding_list = []
        all_embedding_list.extend(all_embedding_list1)
        all_embedding_list.extend(all_embedding_list2)
        all_embedding_list.extend(all_embedding_list3)
        concat_all_embedding_list = tf.concat(all_embedding_list, 1)
        #base_runner.add_trace_variable('concat_all_embedding_list', concat_all_embedding_list)

        with tf.variable_scope("concat1_activation"):
            concat1_embedding = tf.concat(all_embedding_list1, 1)
            concat1_activation = tf.nn.tanh(concat1_embedding)
            base_summary.f_summary_add_to_collection(concat1_activation)
            base_summary.f_summary_add_to_collection_gradient(concat1_activation)
        with tf.variable_scope("concat2_activation"):
            concat2_embedding = tf.concat(all_embedding_list2, 1)
            concat2_activation = tf.nn.tanh(concat2_embedding)
            base_summary.f_summary_add_to_collection(concat2_activation)
            base_summary.f_summary_add_to_collection_gradient(concat2_activation)
        with tf.variable_scope("concat3_activation"):
            concat3_embedding = tf.concat(all_embedding_list3, 1)
            concat3_activation = tf.nn.tanh(concat3_embedding)
            base_summary.f_summary_add_to_collection(concat3_activation)
            base_summary.f_summary_add_to_collection_gradient(concat3_activation)
        with tf.variable_scope("concat_all_embeddings"):
            rnn_embedding.append(concat1_activation)
            rnn_embedding.append(concat2_activation)
            rnn_embedding.append(concat3_activation)

    feature_embedding = tf.concat(rnn_embedding, 1)
    concat_all = tf.concat([dense, feature_embedding], 1)  # [bachsize , (1_dim + 2_dim + deep_dim) * window_size]
    '''
    concat_all = dense
    ## NET WORK STRUCTURE
    input_dim = dense_dim #+ total_embedding_dim
    #inputs = tf.reshape(concat_all, [-1, input_dim])
    #base_runner.add_trace_variable('inputs', inputs)
    batch_size = tf.shape(realsize)[0]
    real_size_list = tf.reshape(realsize, [-1])
    
    layer5_state_size = 32
    layer5_cell = tf.contrib.rnn.BasicRNNCell(layer5_state_size)
    layer5_state = layer5_cell.zero_state(batch_size, tf.float32)
    
    dense_layer1_state_size = 256
    dense_layer1_cell = tf.contrib.rnn.BasicRNNCell(dense_layer1_state_size)
    dense_layer1_state = dense_layer1_cell.zero_state(batch_size, tf.float32)
  
    dense_rnn_inputs = tf.reshape(dense, [-1, stepsize, dense_dim])
    base_runner.add_trace_variable('dense_rnn_inputs', dense_rnn_inputs)
    '''
    with tf.variable_scope("dense-rnn-layer1"):
    #with tf.variable_scope("dense-rnn-layer1", custom_getter=custom_getter):
        dense_layer1_output, dense_layer1_state = tf.nn.dynamic_rnn(dense_layer1_cell, 
    					            dense_rnn_inputs, 
					            sequence_length=real_size_list,
					            initial_state=dense_layer1_state)
        base_summary.f_summary_add_to_collection(dense_layer1_output)
        base_summary.f_summary_add_to_collection(dense_layer1_state)
        base_runner.add_trace_variable('dense_layer1_output', dense_layer1_output)
        dense_layer1_output = tf.reshape(dense_layer1_output, [-1, 256])
    '''
    with tf.variable_scope("layer1", custom_getter=custom_getter):
        dense_layer1_output = full_connect_relu(dense, [dense_dim, 256], [256], False, ps_num=ps_num,
                                          init_val=dense_ini_val, stddev=dense_stddev)
    with tf.variable_scope("layer2", custom_getter=custom_getter):
        #layer2_output = full_connect_relu(tf.concat([dense_layer1_output, sparse_layer1_output], 1), [512, 256], [256], False, ps_num=ps_num,
        #layer2_output = full_connect_relu(tf.concat([dense_layer1_output, sparse_layer1_output], 1), [257, 256], [256], False, ps_num=ps_num,
        layer2_output = full_connect_relu(dense_layer1_output, [256, 256], [256], False, ps_num=ps_num,
                                          init_val=dense_ini_val, stddev=dense_stddev)
        base_summary.f_summary_add_to_collection(layer2_output)
        base_summary.f_summary_add_to_collection_gradient(layer2_output)
        base_runner.add_trace_variable('layer2_output', layer2_output)

    with tf.variable_scope("layer3", custom_getter=custom_getter):
        layer3_output = full_connect_relu(layer2_output, [256, 128], [128], False, ps_num=ps_num, init_val=dense_ini_val,
                                          stddev=dense_stddev)
        base_summary.f_summary_add_to_collection(layer3_output)
        base_summary.f_summary_add_to_collection_gradient(layer3_output)
        base_runner.add_trace_variable('layer3_output', layer3_output)
    
    with tf.variable_scope("layer4", custom_getter=custom_getter):
        layer4_output = full_connect_relu(layer3_output, [128, 64], [64], False, ps_num=ps_num, init_val=dense_ini_val,
                                          stddev=dense_stddev)
        base_summary.f_summary_add_to_collection(layer4_output)
        base_summary.f_summary_add_to_collection_gradient(layer4_output)
        base_runner.add_trace_variable('layer4_output', layer4_output)
        layer4_output = tf.reshape(layer4_output, [-1, stepsize, 64])

    with tf.variable_scope("layer5"):
    #with tf.variable_scope("layer5", custom_getter=custom_getter):
        layer5_output, layer5_state = tf.nn.dynamic_rnn(layer5_cell, 
    					            layer4_output, 
					            sequence_length=real_size_list,
					            initial_state=layer5_state)
        base_summary.f_summary_add_to_collection(layer5_output)
        base_summary.f_summary_add_to_collection_gradient(layer5_output)
        layer5_output = tf.reshape(layer5_output, [-1, 32])
        base_runner.add_trace_variable('layer5_output', layer5_output)

    with tf.variable_scope("layer6", custom_getter=custom_getter):
        layer6_output = full_connect(layer5_output, [32, 1], [1], False, ps_num=ps_num, init_val=dense_ini_val,
                                     stddev=dense_stddev)
        base_summary.f_summary_add_to_collection(layer6_output)
        base_summary.f_summary_add_to_collection_gradient(layer6_output)
        base_runner.add_trace_variable('layer6_output', layer6_output)

    base_summary.f_summary_add_to_collection_gradient(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="embedding_sparse"))
    base_summary.f_summary_add_to_collection_gradient(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="bi_interaction_afm"))
    base_summary.f_summary_add_to_collection_gradient(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="layer6_output"))
    base_summary.f_summary_add_to_collection_gradient(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="layer5_output"))
    base_summary.f_summary_add_to_collection_gradient(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="layer4_output"))
    base_summary.f_summary_add_to_collection_gradient(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="layer3_output"))
    base_summary.f_summary_add_to_collection_gradient(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="layer2_output"))
    base_summary.f_summary_add_to_collection_gradient(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="sparse_layer1_output"))
    base_summary.f_summary_add_to_collection_gradient(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="dense_layer1_output"))
    base_summary.f_summary_add_to_collection(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="embedding_sparse"))
    base_summary.f_summary_add_to_collection(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="bi_interaction_afm"))
    base_summary.f_summary_add_to_collection(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="layer6_output"))
    base_summary.f_summary_add_to_collection(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="layer5_output"))
    base_summary.f_summary_add_to_collection(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="layer4_output"))
    base_summary.f_summary_add_to_collection(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="layer3_output"))
    base_summary.f_summary_add_to_collection(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="layer2_output"))
    base_summary.f_summary_add_to_collection(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="sparse_layer1_output"))
    base_summary.f_summary_add_to_collection(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="dense_layer1_output"))
    return layer6_output, sparse_datas


def get_initializer(init_val=1, dtype=tf.float32, stddev=0.1, value=0.0):
    if init_val == 0:
        return tf.constant_initializer(value=value, dtype=dtype)
    elif init_val == 1:
        return tf.truncated_normal_initializer(dtype=dtype, stddev=stddev)
    elif init_val == 2:
        # factor*[-sqrt(3) / sqrt(dim), sqrt(3) / sqrt(dim)]
        # stddev=factor/sqrt(N)
        # where factor=input stddev!!!!!!!!!!!!!!!
        return tf.uniform_unit_scaling_initializer(factor=stddev, seed=10, dtype=tf.float32)
    else:
        return None


def merged_sparse_embedding(sp_tensors, input_dimensions, embedding_dimension, ps_num, init_val, stddev):
    input_dim = 0
    for i in range(len(input_dimensions)):
        input_dim = input_dim + input_dimensions[i]

    max_fids = [dim - 1 for dim in input_dimensions]
    weights_shape = [input_dim, embedding_dimension]
    if ps_num > 1:
        weights = tf.get_variable("weights",
                                  weights_shape,
                                  initializer=get_initializer(init_val=init_val, dtype=tf.float32, stddev=stddev),
                                  partitioner=tf.min_max_variable_partitioner(max_partitions=ps_num))
    else:
        weights = tf.get_variable("weights",
                                  weights_shape,
                                  initializer=get_initializer(init_val=init_val, dtype=tf.float32, stddev=stddev))

    return merged_embedding_lookup_sparse(weights, sp_tensors, embedding_dimension, max_fids, name='merged_embedding',
                                          combiner='sum')

def step_merged_sparse_embedding(sp_tensors, input_dimensions, embedding_dimension, ps_num, init_val, stddev):
    input_dim = 0
    for i in range(len(input_dimensions)):
        input_dim = input_dim + input_dimensions[i]

    max_fids = [dim - 1 for dim in input_dimensions]
    weights_shape = [input_dim, embedding_dimension]
    if ps_num > 1:
        weights = tf.get_variable("weights",
                                  weights_shape,
                                  initializer=get_initializer(init_val=init_val, dtype=tf.float32, stddev=stddev),
                                  partitioner=tf.min_max_variable_partitioner(max_partitions=ps_num))
    else:
        weights = tf.get_variable("weights",
                                  weights_shape,
                                  initializer=get_initializer(init_val=init_val, dtype=tf.float32, stddev=stddev))
    '''
    result = []
    for i in range(len(sp_tensors)):
        curr_result = []
        for j in range(len(sp_tensors[i])):
            curr_result.append(tf.zeros([1000, embedding_dimension]))
        result.append(curr_result)
    return result
    '''
    return step_merged_embedding_lookup_sparse(weights, sp_tensors, embedding_dimension, max_fids, name='step_merged_embedding',
                                               combiner='sum')

def full_connect(train_inputs, weights_shape, biases_shape, no_biases, ps_num, init_val, stddev):
    # weights
    if ps_num > 1:
        weights = tf.get_variable("weights",
                                  weights_shape,
                                  initializer=get_initializer(init_val=init_val, stddev=stddev),
                                  regularizer=tf.nn.l2_loss,
                                  partitioner=tf.min_max_variable_partitioner(max_partitions=ps_num))
    else:
        weights = tf.get_variable("weights",
                                  weights_shape,
                                  regularizer=tf.nn.l2_loss,
                                  initializer=get_initializer(init_val=init_val, stddev=stddev))

    if no_biases:
        # matmul
        train = tf.matmul(train_inputs, weights)
        print(weights.get_shape())
        return train
    else:
        # biases
        biases = tf.get_variable("biases",
                                 biases_shape,
                                 initializer=get_initializer(init_val=0, value=0.0002))
        print(weights.get_shape())
        # matmul
        train = tf.matmul(train_inputs, weights) + biases
        return train


def full_connect_sparse(train_inputs, weights_shape, biases_shape, no_biases, ps_num, sp_weights, init_val, stddev):
    # weights
    if ps_num > 1:
        weights = tf.get_variable("weights",
                                  weights_shape,
                                  initializer=get_initializer(init_val=init_val, stddev=stddev),
                                  partitioner=tf.min_max_variable_partitioner(max_partitions=ps_num))
    else:
        weights = tf.get_variable("weights",
                                  weights_shape,
                                  initializer=get_initializer(init_val=init_val, stddev=stddev))

    train = tf.nn.embedding_lookup_sparse(weights, sp_ids=train_inputs, sp_weights=sp_weights, combiner="sum")
    if not no_biases:
        # biases
        biases = tf.get_variable("biases",
                                 biases_shape,
                                 initializer=get_initializer(init_val=init_val, stddev=stddev))
        train = train + biases
    return train


def full_connect_sparse_relu(train_inputs, weights_shape, biases_shape, no_biases, ps_num, sp_weights, init_val,
                             stddev):
    train = full_connect_sparse(train_inputs, weights_shape, biases_shape, no_biases, ps_num, sp_weights,
                                init_val=init_val, stddev=stddev)
    return tf.nn.relu(train)


# relu full connect
def full_connect_relu(train_inputs, weights_shape, biases_shape, no_biases, ps_num, init_val, stddev):
    train = full_connect(train_inputs, weights_shape, biases_shape, no_biases, ps_num, init_val=init_val, stddev=stddev)
    return tf.nn.relu(train)


def loss(logits, labels, mask, weight_decay):
    with tf.variable_scope("loss"):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
	pos_weight = 1/3
        cross_entropy = 3 * tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits,
                                                                     pos_weight=pos_weight,
                                                                     name=None)
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    return tf.reduce_mean(cross_entropy, name='xentropy_mean') + weight_decay * tf.add_n(reg_losses)


def auc(predictions, labels, idx, num_thresholds=200):
    with tf.variable_scope("auc"):
        predictions = tf.boolean_mask(tf.reshape(predictions, [-1]), tf.reshape(idx, [-1]))
        labels = tf.boolean_mask(tf.reshape(labels, [-1]), tf.reshape(idx, [-1]))
        _, auc_op = tf.contrib.metrics.streaming_auc(predictions, labels, num_thresholds=num_thresholds)
    return auc_op


def bi_interaction_fmwffm(attr_fs,ps_num,init_val,biscopename, sampleIdx):
    # Attentional Factorization Machines Learning the Weight of Feature Interactions via Attention Networks
    if sampleIdx == 0:
        with tf.variable_scope(biscopename) as scope:
            product_list=[]
            dim_product=0
            for i in range(len(attr_fs)):
                for j in range(len(attr_fs)):
                    if i < j:
                        for x1 in attr_fs[i]:
                            for x2 in attr_fs[j]:
                                scopename = "bi_interaction"
                                with tf.variable_scope(scopename):
                                    x=tf.multiply(x1, x2, name="element_multiply")
                                    product_list.append(x)
                                    base_summary.f_summary_add_to_collection(x)
                                    dim_product=dim_product+1
            winival=2.0/dim_product
            weights = tf.get_variable("weights",
                                  [dim_product,1],
                                              initializer=get_initializer(init_val=init_val,  value=winival),
                                  regularizer = tf.nn.l2_loss,
                                  partitioner = tf.min_max_variable_partitioner(max_partitions=ps_num))
            attention_list=tf.split(weights,dim_product,0)
            bi_list=tf.multiply(attention_list,product_list)
            bi_outfml = tf.reduce_sum(product_list, 0)
            bi_outwfm=tf.reduce_sum(bi_list,0)
    else:
        with tf.variable_scope(biscopename, reuse=True) as scope:
            product_list=[]
            dim_product=0
            for i in range(len(attr_fs)):
                for j in range(len(attr_fs)):
                    if i < j:
                        for x1 in attr_fs[i]:
                            for x2 in attr_fs[j]:
                                scopename = "bi_interaction"
                                with tf.variable_scope(scopename):
                                    x=tf.multiply(x1, x2, name="element_multiply")
                                    product_list.append(x)
                                    base_summary.f_summary_add_to_collection(x)
                                    dim_product=dim_product+1
            winival=2.0/dim_product
            weights = tf.get_variable("weights",
                                  [dim_product,1],
                                              initializer=get_initializer(init_val=init_val,  value=winival),
                                  regularizer = tf.nn.l2_loss,
                                  partitioner = tf.min_max_variable_partitioner(max_partitions=ps_num))
            attention_list=tf.split(weights,dim_product,0)
            bi_list=tf.multiply(attention_list,product_list)
            bi_outfml = tf.reduce_sum(product_list, 0)
            bi_outwfm=tf.reduce_sum(bi_list,0)
    return bi_outfml,bi_outwfm
