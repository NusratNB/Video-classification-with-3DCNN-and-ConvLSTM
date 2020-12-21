from keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling3D, Dense
from keras.layers import MaxPooling2D, MaxPooling3D
from keras.layers import Conv3D, Input, Bidirectional
from keras.models import Model
from keras.layers import TimeDistributed, Activation, ConvLSTM2D


class Model_3d(object):
    '''
    Implementation of the
    https://openaccess.thecvf.com/content_ICCV_2017_workshops/w44/html/Zhang_Learning_Spatiotemporal_Features_ICCV_2017_paper.html
    paper
    '''

    def __init__(self, number_fo_frames, width, height, channel):
        self.width = width
        self.height = height
        self.channel = channel
        self.nb_frames = number_fo_frames

    def cnn3d(self, input_shape):
        first_layer = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), name='conv3d_1')(input_shape)
        bch_first = BatchNormalization()(first_layer)
        act_1 = Activation("relu")(bch_first)
        pooling_first = MaxPooling3D((1, 2, 2), strides=(1, 2, 2))(act_1)

        second_layer = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), name='conv3d_2')(pooling_first)
        bch_sec = BatchNormalization()(second_layer)
        act_2 = Activation("relu")(bch_sec)
        pooling_sec = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(act_2)

        third_layer = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), name='conv3d_3a')(pooling_sec)

        fourth_layer = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), name='conv3d_3b')(third_layer)
        bch_fourth = BatchNormalization()(fourth_layer)
        act_4 = Activation("relu")(bch_fourth)

        return act_4

    def convlstm(self, layer):
        first_conv_lstm = Bidirectional(ConvLSTM2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                                                   activation="relu", return_sequences=True))(layer)
        second_conv_lstm = Bidirectional(ConvLSTM2D(256, kernel_size=(3, 3), strides=(1, 1),
                                                    activation="relu", return_sequences=True))(first_conv_lstm)
        return second_conv_lstm

    def cnn2d(self, input_layer):
        first_layer = TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), name="conv2d_1"))(
            input_layer)
        bch_first = TimeDistributed(BatchNormalization())(first_layer)
        act_1 = TimeDistributed(Activation("relu"))(bch_first)
        pooling_first = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(act_1)

        second_layer = TimeDistributed(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), name="conv2d_2"))(
            pooling_first)
        bch_sec = TimeDistributed(BatchNormalization())(second_layer)
        act_2 = TimeDistributed(Activation("relu"))(bch_sec)
        pooling_sec = TimeDistributed(MaxPooling2D((3, 3), strides=(1, 1)))(act_2)

        third_layer = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), name="conv2d_3"))(
            pooling_sec)
        bch_third = TimeDistributed(BatchNormalization())(third_layer)
        act_3 = TimeDistributed(Activation("relu"))(bch_third)
        pooling_third = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2)))(act_3)

        return pooling_third

    def get_model(self):
        shape = (self.nb_frames, self.width, self.height, self.channel)
        video_input = Input(shape=shape)
        cnn3d_part = self.cnn3d(video_input)
        convlstm_part = self.convlstm(cnn3d_part)
        cnn2d_part = self.cnn2d(convlstm_part)
        flatten3d = GlobalAveragePooling3D()(cnn2d_part)
        pre_final = Dense(64)(flatten3d)

        act_relu = Activation("relu")(pre_final)
        final = Dense(2)(act_relu)
        output = Activation('softmax')(final)
        model = Model(video_input, output)

        model.summary()
        return model



