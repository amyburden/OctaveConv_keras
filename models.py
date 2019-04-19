from keras.layers import *
from keras.models import *
import keras.backend as K

def Act(act_type, name):
    if act_type == 'prelu':
        body = PReLU(name=name, shared_axes=[1, 2])
    elif act_type == 'leaky':
        body = LeakyReLU(0.125, name=name + 'l')
    else:
        body = Activation(act_type, name=name)
    return body
    
def Residual_Unit_last(hf_data, lf_data, alpha, num_in, num_mid, num_out, name, first_block=False, stride=(1, 1), **kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    act_type = kwargs.get('version_act', 'leaky')
    wd = kwargs.get('wd', 0.00004)
    epsilon = kwargs.get('epsilon', 2e-5)
    name_base = ('%s_conv-' % name)
    
    hf_data_1, lf_data_1 = OctConv(hf_data, lf_data, (alpha,alpha),num_in, num_mid,kernel=(1,1), name=('%sm1' % name_base))
    hf_data_1 = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + 'm1_hf__bn')(hf_data_1)
    hf_data_1 = Act(act_type=act_type, name=name_base +'m1_hf__'+act_type)(hf_data_1)
    lf_data_1 = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + 'm1_lf__bn')(lf_data_1)
    lf_data_1 = Act(act_type=act_type, name=name_base +'m1_lf__'+act_type)(lf_data_1)
    print ('data_1:', hf_data_1._keras_shape, lf_data_1._keras_shape)
    hf_data_2, lf_data_2 = OctConv(hf_data_1, lf_data_1, (alpha,0),num_mid, num_mid,kernel=(3,3), stride=stride, padding=1, name=('%sm2' % name_base))
    hf_data_2 = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + 'm2__bn')(hf_data_2)
    hf_data_2 = Act(act_type=act_type, name=name_base +'m2__'+act_type)(hf_data_2)
    print ('data_2:', hf_data_2._keras_shape)
    conv_m3 = Conv2D(num_out, (1,1), name=('%sm3' % name_base))(hf_data_2)
    conv_m3 = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + 'm3__bn')(conv_m3)
    print ('data_3:', conv_m3._keras_shape)
    
    if first_block:
        hf_data_sc, lf_data_sc = OctConv(hf_data, lf_data, (alpha,0),num_mid, num_out, kernel=(1,1),stride=stride, name=('%ssc' % name_base))
        hf_data_sc = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + 'sc__bn')(hf_data_sc)
    else:
        hf_data_sc = hf_data
    
    hf_outputs = Add()([conv_m3, hf_data_sc])
    hf_outputs = Act(act_type=act_type, name=name +'_act__'+act_type)(hf_outputs)

    return hf_outputs

def Residual_Unit_norm(data, num_in, num_mid, num_out, name, first_block=False, stride=(1, 1), **kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    act_type = kwargs.get('version_act', 'leaky')
    wd = kwargs.get('wd', 0.00004)
    epsilon = kwargs.get('epsilon', 2e-5)
    name_base = ('%s_conv-' % name)
    
    x = Conv2D(num_mid, (1,1), name='%s_conv-m1' % name)(data)
    x = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name='%s_conv-m1_bn' % name)(x)
    x = Act(act_type=act_type, name='%s_conv-m1_relu' % name)(x)
    print ('data_1:', x._keras_shape)
    x = ZeroPadding2D(((1, 1), (1, 1)))(x)
    x = Conv2D(num_mid, (3,3), strides=stride, name='%s_conv-m2' % name)(x)
    x = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name='%s_conv-m2_bn' % name)(x)
    x = Act(act_type=act_type, name=('%s_conv-m2_relu' % name))(x)
    print ('data_2:', x._keras_shape)
    x = Conv2D(num_out, (1,1), name='%s_conv-m3' % name)(x)
    x = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name='%s_conv-m3_bn' % name)(x)
    print ('data_3:', x._keras_shape)
    if first_block:
        conv1sc = Conv2D(num_out, kernel_size=(1, 1), strides=stride, use_bias=False,
                         name='%s_conv-sc' % name)(data)
        sc = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + '_bnsc')(conv1sc)
    else:
        sc = data
    
    x = Add()([x, sc])
    x = Act(act_type=act_type, name=name +'_act__'+act_type)(x)

    return x

def Residual_Unit(hf_data, lf_data, alpha, num_in, num_mid, num_out, name, first_block=False, stride=(1, 1), **kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    act_type = kwargs.get('version_act', 'leaky')
    wd = kwargs.get('wd', 0.00004)
    epsilon = kwargs.get('epsilon', 2e-5)
    name_base = ('%s_conv-' % name)
    
    hf_data_1, lf_data_1 = OctConv(hf_data, lf_data, (alpha,alpha),num_in, num_mid,kernel=(1,1), name=('%sm1' % name_base))
    hf_data_1 = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + 'm1_hf__bn')(hf_data_1)
    hf_data_1 = Act(act_type=act_type, name=name_base +'m1_hf__'+act_type)(hf_data_1)
    lf_data_1 = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + 'm1_lf__bn')(lf_data_1)
    lf_data_1 = Act(act_type=act_type, name=name_base +'m1_lf__'+act_type)(lf_data_1)
    print ('data_1:', hf_data_1._keras_shape, lf_data_1._keras_shape)
    hf_data_2, lf_data_2 = OctConv(hf_data_1, lf_data_1, (alpha,alpha),num_mid, num_mid,kernel=(3,3), stride=stride, padding=1, name=('%sm2' % name_base))
    hf_data_2 = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + 'm2_hf__bn')(hf_data_2)
    hf_data_2 = Act(act_type=act_type, name=name_base +'m2_hf__'+act_type)(hf_data_2)
    lf_data_2 = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + 'm2_lf__bn')(lf_data_2)
    lf_data_2 = Act(act_type=act_type, name=name_base +'m2_lf__'+act_type)(lf_data_2)
    print ('data_2:', hf_data_2._keras_shape, lf_data_2._keras_shape)
    hf_data_3, lf_data_3 = OctConv(hf_data_2, lf_data_2, (alpha,alpha),num_mid, num_out,kernel=(1,1), name=('%sm3' % name_base))
    hf_data_3 = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + 'm3_hf__bn')(hf_data_3)
    lf_data_3 = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + 'm3_lf__bn')(lf_data_3)
    print ('data_3:', hf_data_3._keras_shape, lf_data_3._keras_shape)
    if first_block:
        hf_data_sc, lf_data_sc = OctConv(hf_data, lf_data, (alpha,alpha),num_mid, num_out, kernel=(1,1),stride=stride, name=('%ssc' % name_base))
        hf_data_sc = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + 'sc_hf__bn')(hf_data_sc)
        lf_data_sc = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name=name_base + 'sc_lf__bn')(lf_data_sc)
    else:
        hf_data_sc = hf_data
        lf_data_sc = lf_data
    
    hf_outputs = Add()([hf_data_3, hf_data_sc])
    hf_outputs = Act(act_type=act_type, name=name +'_hf_act__'+act_type)(hf_outputs)
    lf_outputs = Add()([lf_data_3, lf_data_sc])
    lf_outputs = Act(act_type=act_type, name=name +'_lf_act__'+act_type)(lf_outputs)

    return hf_outputs, lf_outputs

def OctConv(hf_data, lf_data, settings, ch_in, ch_out, name, kernel=(1,1), padding=0, stride=(1,1)):
    alpha_in, alpha_out = settings
    hf_ch_in = int(ch_in * (1 - alpha_in))
    hf_ch_out = int(ch_out * (1 - alpha_out))

    lf_ch_in = ch_in - hf_ch_in
    lf_ch_out = ch_out - hf_ch_out

    if stride == (2, 2):
        hf_data = AveragePooling2D((2,2), name=('%s_hf_down' % name))(hf_data)
#         hf_data = mx.symbol.Pooling(data=hf_data, pool_type='avg', kernel=(2,2), stride=(2,2), name=('%s_hf_down' % name))
    if padding > 0:
        hf_data_pad = ZeroPadding2D(((padding, padding), (padding, padding)))(hf_data)
    else:
        hf_data_pad = hf_data
    hf_conv = Conv2D(hf_ch_out, kernel, name=('%s_hf_conv' % name))(hf_data_pad)
    
    hf_pool = AveragePooling2D((2,2), name=('%s_hf_pool' % name))(hf_data)
    if lf_ch_out > 0:
        if padding > 0:
            hf_pool_pad = ZeroPadding2D(((padding, padding), (padding, padding)))(hf_pool)
        else:
            hf_pool_pad = hf_pool
        hf_pool_conv = Conv2D(lf_ch_out, kernel, name=('%s_hf_pool_conv' % name))(hf_pool_pad)
    else:
        hf_pool_conv = None
        
    if lf_data is not None:
        if padding > 0:
            lf_data_pad = ZeroPadding2D(((padding, padding), (padding, padding)))(lf_data)
        else:
            lf_data_pad = lf_data
        lf_conv = Conv2D(hf_ch_out, kernel, name=('%s_lf_conv' % name))(lf_data_pad)

        if stride == (2, 2):
            lf_upsample = lf_conv
            lf_down = AveragePooling2D((2,2), name=('%s_lf_down' % name))(lf_data)
        else:
            lf_upsample = UpSampling2D((2,2), name='%s_lf_upsample' % name)(lf_conv)
            lf_down = lf_data
        if lf_ch_out > 0:
            if padding > 0:
                lf_down_pad = ZeroPadding2D(((padding, padding), (padding, padding)))(lf_down)
            else:
                lf_down_pad = lf_down
            lf_down_conv = Conv2D(lf_ch_out, kernel, name=('%s_lf_down_conv' % name))(lf_down_pad)
            out_l = Add()([hf_pool_conv, lf_down_conv])
        else:
            out_l = None
            
        out_h = Add()([hf_conv, lf_upsample])
    else:
        out_h = hf_conv
        out_l = hf_pool_conv
    return out_h, out_l

def base_model(input_shape=(224, 224, 3), weights=None, bottleneck=256, alpha=0.5, **kwargs):
    num_layers = kwargs.get('num_layers', 28)
    act_type = kwargs.get('version_act', 'leaky')
    bn_mom = kwargs.get('bn_mom', 0.9)
    wd = kwargs.get('wd', 0.00004)
    wd_mult = kwargs.get('wd_mult', 10.)
    ft_mult = kwargs.get('ft_mult', 1.)
    epsilon = kwargs.get('epsilon', 2e-5)
    
    num_stages = 4
    if num_layers > 101:
        filter_list = [128*ft_mult, 512*ft_mult, 1024*ft_mult, 2048*ft_mult, 4096*ft_mult]
    else:
        filter_list = [64*ft_mult, 256*ft_mult, 512*ft_mult, 1024*ft_mult, 2048*ft_mult]
    
    filter_list = map(int, filter_list)
    if num_layers == 18:
        blocks = [2, 2, 2, 2]
    elif num_layers == 28:
        blocks = [3, 4, 3, 3]
    elif num_layers == 34:
        blocks = [3, 4, 6, 3]
    elif num_layers == 50:
        blocks = [3, 4, 14, 3]
    elif num_layers == 100:
        blocks = [3, 13, 30, 3]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
    
    input_img = Input(input_shape)
    # conv1
    body = Conv2D(filter_list[0], 7, strides=2, padding="same", use_bias=False, kernel_regularizer=l2(wd),
                  name='conv0')(input_img)
    body = BatchNormalization(epsilon=epsilon, momentum=bn_mom, name='bn0')(body)
    body = Act(act_type=act_type, name='relu0')(body)
    body = MaxPool2D((3,3),strides=(2,2), padding='same', name='pool1')(body)
    # conv2
    num_in =  32
    num_mid = 64
    num_out = 256
    
    hf_conv1_x = body
    lf_conv1_x = None
    for i in range(1, blocks[0]+1):
        print 'block1:', i
        hf_conv2_x, lf_conv2_x = Residual_Unit( 
                                 hf_data = (hf_conv1_x if i == 1 else hf_conv2_x),
                                 lf_data = (lf_conv1_x if i == 1 else lf_conv2_x),
                                 alpha=alpha,
                                 num_in = (num_in if i == 1 else num_out),   
                                 num_mid = num_mid,
                                 num_out = num_out, 
                                 name = ('conv2_B%02d'% i),
                                 first_block = (i==1), 
                                 stride = ((1, 1) if (i == 1) else (1,1)) )
    
    # conv3
    num_in =  num_out
    num_mid = int(num_mid*2)
    num_out = int(num_out*2)
    for i in range(1, blocks[1]+1):
        print 'block2:', i
        hf_conv3_x, lf_conv3_x = Residual_Unit( 
                                 hf_data = (hf_conv2_x if i == 1 else hf_conv3_x),
                                 lf_data = (lf_conv2_x if i == 1 else lf_conv3_x),
                                 alpha=alpha,
                                 num_in = (num_in if i == 1 else num_out),   
                                 num_mid = num_mid,
                                 num_out = num_out, 
                                 name = ('conv3_B%02d'% i),
                                 first_block = (i==1), 
                                 stride = ((2, 2) if (i == 1) else (1,1)))
        
    # conv4
    num_in =  num_out
    num_mid = int(num_mid*2)
    num_out = int(num_out*2)
    for i in range(1,  blocks[2]+1):
        print 'block3:', i
        hf_conv4_x, lf_conv4_x = Residual_Unit( 
                                 hf_data = (hf_conv3_x if i == 1 else hf_conv4_x),
                                 lf_data = (lf_conv3_x if i == 1 else lf_conv4_x),
                                 alpha=alpha,
                                 num_in = (num_in if i == 1 else num_out),   
                                 num_mid = num_mid,
                                 num_out = num_out, 
                                 name = ('conv4_B%02d'% i),
                                 first_block = (i==1), 
                                 stride = ((2, 2) if (i == 1) else (1,1)) )
    
    # conv5
    num_in =  num_out
    num_mid = int(num_mid*2)
    num_out = int(num_out*2)
    i = 1
    print 'block4:', 1
    conv5_x = Residual_Unit_last( 
                             hf_data = (hf_conv4_x if i == 1 else hf_conv5_x),
                             lf_data = (lf_conv4_x if i == 1 else lf_conv5_x),
                             alpha=alpha,
                             num_in = (num_in if i == 1 else num_out),   
                             num_mid = num_mid,
                             num_out = num_out, 
                             name = ('conv5_B%02d'% i),
                             first_block = (i==1), 
                             stride = ((2, 2) if (i == 1) else (1,1)))
    for i in range(2, blocks[3]+1):
        print 'block4:', i
        conv5_x = Residual_Unit_norm(data = conv5_x,
                                 num_in = (num_in if i == 1 else num_out),   
                                 num_mid = num_mid,
                                 num_out = num_out, 
                                 name = ('conv5_B%02d'% i),
                                 first_block = (i==1), 
                                 stride = ((2, 2) if (i == 1) else (1,1)))
    m = Model(input_img, conv5_x)
    if weights:
        m.load_weights(weights)
    return m
    
