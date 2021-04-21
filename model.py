import tensorflow.compat.v1 as tf

def convolution(input, shape, stride = [1, 1, 1, 1], use_bias = False, kv_name = "weights", bv_name = "bias"):
    sh_in = input.get_shape().as_list()
    shape = [shape[0], shape[1], sh_in[3], shape[2]]
    kernel = tf.get_variable(kv_name, shape, dtype=tf.float32)
    convolution = tf.nn.conv2d(input, kernel, stride, padding='SAME')
    if use_bias:
        bias = tf.get_variable(bv_name, [shape[3]], dtype=tf.float32)
        convolution = tf.nn.bias_add(convolution, bias)

    return convolution

def batchNormalisation(input, sv_name = "scale", ov_name = "offset", mv_name = "mean", vv_name = "variance"):
    shape = input.get_shape().as_list()
    n_channels = shape[3]
    scale = tf.get_variable(sv_name, [n_channels], dtype=tf.float32)
    offset = tf.get_variable(ov_name, [n_channels], dtype=tf.float32)
    mean = tf.get_variable(mv_name, [n_channels], dtype=tf.float32)
    variance = tf.get_variable(vv_name, [n_channels], dtype=tf.float32)
    out = tf.nn.batch_normalization(input, mean, variance, offset, scale, 0.00000001)

    return out

def concat_pad(upConvolution, resnetBlock):
    shape1 = resnetBlock.get_shape().as_list()[1:-1]
    shape2 = upConvolution.get_shape().as_list()[1:-1]
    padding = [a_i - b_i for a_i, b_i in zip(shape2, shape1)]
    block_padded = tf.pad(resnetBlock, [[0, 0], [0, padding[0]], [0, padding[1]], [0, 0]])
    res = tf.concat([upConvolution, block_padded], 3)
    return res

def upConvolution(input, noOfOutputChannels):
    conv1 = convolution(input, [3, 3, noOfOutputChannels], use_bias=True, kv_name="weights1", bv_name="bias1")
    conv2 = convolution(input, [2, 3, noOfOutputChannels], use_bias=True, kv_name="weights2", bv_name="bias2")
    conv3 = convolution(input, [3, 2, noOfOutputChannels], use_bias=True, kv_name="weights3", bv_name="bias3")
    conv4 = convolution(input, [2, 2, noOfOutputChannels], use_bias=True, kv_name="weights4", bv_name="bias4")

    sh = conv1.get_shape().as_list()
    dim = len(sh[1:-1])
    tmp1 = tf.reshape(conv1, [-1] + sh[-dim:])
    tmp2 = tf.reshape(conv2, [-1] + sh[-dim:])
    tmp3 = tf.reshape(conv3, [-1] + sh[-dim:])
    tmp4 = tf.reshape(conv4, [-1] + sh[-dim:])

    concat1 = tf.concat([tmp1, tmp3], 2)
    concat2 = tf.concat([tmp2, tmp4], 2)
    
    concat_final = tf.concat([concat1, concat2], 1)
    out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
    res = tf.reshape(concat_final, out_size)

    return res

def upProjection(input, noOfOutputChannels, variable_scope):
    
    with tf.variable_scope(variable_scope):
        with tf.variable_scope('fast1_c'):
            conv1 = upConvolution(input, noOfOutputChannels)
        conv_bn1 = tf.nn.relu(batchNormalisation(conv1, vv_name="variance1", ov_name="batchnorm_offset1", mv_name="mean1", sv_name="batchnorm_scale1"))

        conv2 = convolution(conv_bn1, [3, 3, noOfOutputChannels], kv_name="weights2")
        conv_bn2 = batchNormalisation(conv2, vv_name="variance2", ov_name="batchnorm_offset2", mv_name="mean2", sv_name="batchnorm_scale2")

        with tf.variable_scope('fast2_c'):
            conv3 = upConvolution(input, noOfOutputChannels)
        conv_bn3 = batchNormalisation(conv3, vv_name="variance3", ov_name="batchnorm_offset3", mv_name="mean3", sv_name="batchnorm_scale3")

        sum = conv_bn2 + conv_bn3
        res = tf.nn.relu(sum)

    return res

def resnetBlock(input, nfilter1, nfilter2, use_shortcut=False, stride=1, variable_scope=""):
    
    with tf.variable_scope(variable_scope):
        # Conv1
        conv1 = convolution(input, [1, 1, nfilter1], [1, stride, stride, 1], kv_name="weights1")
        conv1_bn = batchNormalisation(conv1, vv_name="variance1", ov_name="batchnorm_offset1", mv_name="mean1", sv_name="batchnorm_scale1")
        conv1_relu = tf.nn.relu(conv1_bn)

        # Conv2
        conv2 = convolution(conv1_relu, [3, 3, nfilter1], kv_name="weights2")
        conv2_bn = batchNormalisation(conv2, vv_name="variance2", ov_name="batchnorm_offset2", mv_name="mean2", sv_name="batchnorm_scale2")
        conv2_relu = tf.nn.relu(conv2_bn)

        # Convolve
        conv3 = convolution(conv2_relu, [1, 1, nfilter2], kv_name="weights3")
        conv3_bn = batchNormalisation(conv3, vv_name="variance3", ov_name="batchnorm_offset3", mv_name="mean3", sv_name="batchnorm_scale3")

        if use_shortcut:
            # Conv4
            conv4 = convolution(input, [1, 1, nfilter2], [1, stride, stride, 1], kv_name="weights4")
            conv4_bn = batchNormalisation(conv4, vv_name="variance4", ov_name="batchnorm_offset4", mv_name="mean4", sv_name="batchnorm_scale4")

            # Sum streams up and apply ReLU
            res = tf.nn.relu(tf.add(conv4_bn, conv3_bn))
        else:
            res = tf.nn.relu(tf.add(input, conv3_bn))

    return res

def inference(images):

    #Encoder
    with tf.variable_scope("conv1"):
        conv1 = convolution(images, [7, 7, 64], [1, 2, 2, 1])
        conv1 = batchNormalisation(conv1, vv_name="variance", ov_name="batchnorm_offset", mv_name="mean", sv_name="batchnorm_scale")
        conv1 = tf.nn.relu(conv1)

    pool1 = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

    block1 = resnetBlock(pool1, 64, 256, True, 1, 'block1')
    block2 = resnetBlock(block1, 64, 256, variable_scope='block2')
    block3 = resnetBlock(block2, 64, 256, variable_scope='block3')

    # Scale x * 2
    block4 = resnetBlock(block3, 128, 512, True, 2, 'block4')
    block5 = resnetBlock(block4, 128, 512, variable_scope='block5')
    block6 = resnetBlock(block5, 128, 512, variable_scope='block6')
    block7 = resnetBlock(block6, 128, 512, variable_scope='block7')

    # Scale x * 2
    block8 = resnetBlock(block7, 256, 1024, True, 2, 'block8')
    block9 = resnetBlock(block8, 256, 1024, variable_scope='block9')
    block10 = resnetBlock(block9, 256, 1024, variable_scope='block10')
    block11 = resnetBlock(block10, 256, 1024, variable_scope='block11')
    block12 = resnetBlock(block11, 256, 1024, variable_scope='block12')
    block13 = resnetBlock(block12, 256, 1024, variable_scope='block13')

    # Scale x * 2
    block14 = resnetBlock(block13, 512, 2048, True, 2, 'block14')
    block15 = resnetBlock(block14, 512, 2048, variable_scope='block15')
    block16 = resnetBlock(block15, 512, 2048, variable_scope='block16')
    # End residual units

    with tf.variable_scope("conv2"):
        conv2 = convolution(block16, [1, 1, 1024])
        conv2 = batchNormalisation(conv2, vv_name="variance", ov_name="batchnorm_offset", mv_name="mean", sv_name="batchnorm_scale")
    # End encoder

    # Start decoder
    upproject1 = upProjection(conv2, 512, 'upproject1')
    upproject1 = concat_pad(upproject1, block13)

    upproject2 = upProjection(upproject1, 256, 'upproject2')
    upproject2 = concat_pad(upproject2, block7)

    upproject3 = upProjection(upproject2, 128, 'upproject3')
    upproject3 = concat_pad(upproject3, block3)

    upproject4 = upProjection(upproject3, 64, 'upproject4')

    with tf.variable_scope("conv3"):
        prediction = convolution(upproject4, [3, 3, 1], use_bias = True)
    
    #End decoder

    return prediction
