from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, Input, Conv2DTranspose, AveragePooling2D, Concatenate
from keras.models import Model
from keras.backend import int_shape, is_keras_tensor
import keras.backend as K


class LinkNet:
    """LinkNet architecture.

    The model follows the architecture presented in: https://arxiv.org/abs/1707.03718

    Args:
        num_classes (int): the number of classes to segment.
        input_tensor (tensor, optional): Keras tensor
            (i.e. output of `layers.Input()`) to use as image input for
            the model. Default: None.
        input_shape (tuple, optional): Shape tuple of the model input.
            Default: None.
        initial_block_filters (int, optional): The number of filters after
            the initial block (see the paper for details on the initial
            block). Default: None.
        bias (bool, optional): If ``True``, adds a learnable bias.
            Default: ``False``.

    """

    def __init__(
        self,
        num_classes,
        input_tensor=None,
        input_shape=None,
        initial_block_filters=64,
        bias=False,
        name='linknet'
    ):
        self.num_classes = num_classes
        self.initial_block_filters = initial_block_filters
        self.bias = bias
        self.output_shape = input_shape[:-1] + (num_classes, )

        # Create a Keras tensor from the input_shape/input_tensor
        if input_tensor is None:
            self.input = Input(shape=input_shape, name='input_img')
        elif is_keras_tensor(input_tensor):
            self.input = input_tensor
        else:
            # input_tensor is a tensor but not one from Keras
            self.input = Input(
                tensor=input_tensor, shape=input_shape, name='input_img'
            )

        self.name = name

    def get_model(
        self,
        pretrained_encoder=False,
        weights_path='./checkpoints/linknet_encoder_weights.h5',
        use_shortcuts=True
    ):
        """Initializes a LinkNet model.

        Returns:
            A Keras model instance.

        """
        # Build encoder
        encoder_model = self.get_encoder()
        if pretrained_encoder:
            encoder_model.load_weights(weights_path)
        encoder_out = encoder_model(self.input)

        # Build decoder
        decoder_model = self.get_decoder(encoder_out, use_shortcuts=use_shortcuts)
        decoder_out = decoder_model(encoder_out[:-1])

        return Model(inputs=self.input, outputs=decoder_out, name=self.name)

    def get_encoder(self, name='encoder'):
        """Builds the encoder of a LinkNet architecture.

        Args:
            name (string, optional): The encoder model name.
                Default: 'encoder'.

        Returns:
            The encoder as a Keras model instance.

        """
        # Initial block
        initial1 = Conv2D(
            self.initial_block_filters,
            kernel_size=7,
            strides=2,
            padding='same',
            use_bias=self.bias,
            name=name + '/0/conv2d_1'
        )(self.input)
        initial1 = BatchNormalization(name=name + '/0/bn_1')(initial1)
        initial1 = Activation('relu', name=name + '/0/relu_1')(initial1)
        initial2 = MaxPooling2D(pool_size=2, name=name + '/0/maxpool_1')(initial1)  # yapf: disable

        blocks = [5, 7, 10, 12]
        # Encoder blocks
        encoder1 = self.dense_block(initial2, blocks[0], name='db1')

        encoder2 = self.transition_block(encoder1, 0.5, name='pool2')
        encoder2 = self.dense_block(encoder2, blocks[1], name='db2')

        encoder3 = self.transition_block(encoder2, 0.5, name='pool3')
        encoder3 = self.dense_block(encoder3, blocks[2], name='db3')

        encoder4 = self.transition_block(encoder3, 0.5, name='pool4')
        encoder4 = self.dense_block(encoder4, blocks[3], name='db4')

        return Model(
            inputs=self.input,
            outputs=[
                encoder4, encoder3, encoder2, encoder1, initial2, initial1
            ],
            name=name
        )

    def get_decoder(self, inputs, name='decoder', use_shortcuts=True):
        """Builds the decoder of a LinkNet architecture.

        Args:
            name (string, optional): The encoder model name.
                Default: 'decoder'.

        Returns:
            The decoder as a Keras model instance.

        """
        # Decoder inputs
        encoder4 = Input(shape=int_shape(inputs[0])[1:], name='encoder4')
        encoder3 = Input(shape=int_shape(inputs[1])[1:], name='encoder3')
        encoder2 = Input(shape=int_shape(inputs[2])[1:], name='encoder2')
        encoder1 = Input(shape=int_shape(inputs[3])[1:], name='encoder1')
        initial2 = Input(shape=int_shape(inputs[4])[1:], name='initial2')

        # Decoder blocks
        decoder4 = self.decoder_block(
            encoder4,
            134,
            strides=2,
            bias=self.bias,
            name=name + '/4'
        )
        if use_shortcuts:
            decoder4 = Add(name=name + '/shortcut_e3_d4')([encoder3, decoder4])

        decoder3 = self.decoder_block(
            decoder4,
            108,
            strides=2,
            bias=self.bias,
            name=name + '/3'
        )
        if use_shortcuts:
            decoder3 = Add(name=name + '/shortcut_e2_d3')([encoder2, decoder3])

        decoder2 = self.decoder_block(
            decoder3,
            104,
            strides=2,
            bias=self.bias,
            name=name + '/2'
        )
        if use_shortcuts:
            decoder2 = Add(name=name + '/shortcut_e1_d2')([encoder1, decoder2])

        decoder1 = self.decoder_block(
            decoder2,
            self.initial_block_filters,
            strides=1,
            bias=self.bias,
            name=name + '/1'
        )
        if use_shortcuts:
            decoder1 = Add(name=name + '/shortcut_init_d1')([initial2, decoder1])

        # Final block
        final = Conv2DTranspose(
            self.initial_block_filters // 2,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=self.bias,
            name=name + '/0/transposed2d_1'
        )(decoder1)
        final = BatchNormalization(name=name + '/0/bn_1')(final)
        final = Activation('relu', name=name + '/0/relu_1')(final)

        final = Conv2D(
            self.initial_block_filters // 2,
            kernel_size=3,
            padding='same',
            use_bias=self.bias,
            name=name + '/0/conv2d_1'
        )(final)
        final = BatchNormalization(name=name + '/0/bn_2')(final)
        final = Activation('relu', name=name + '/0/relu_2')(final)

        logits = Conv2DTranspose(
            self.num_classes,
            kernel_size=2,
            strides=2,
            padding='same',
            use_bias=self.bias,
            name=name + '/0/transposed2d_2'
        )(final)

        prediction = Activation('sigmoid')(logits)

        return Model(
            inputs=[
                encoder4, encoder3, encoder2, encoder1, initial2
            ],
            outputs=prediction,
            name=name
        )

    def decoder_block(
        self,
        input,
        out_filters,
        kernel_size=3,
        strides=2,
        projection_ratio=4,
        padding='same',
        bias=False,
        name=''
    ):
        """Creates a decoder block.

        Decoder block architecture:
        1. Conv2D
        2. BatchNormalization
        3. ReLU
        4. Conv2DTranspose
        5. BatchNormalization
        6. ReLU
        7. Conv2D
        8. BatchNormalization
        9. ReLU

        Args:
            input (tensor): A tensor or variable.
            out_filters (int): The number of filters in the block output.
            kernel_size (int, tuple, list, optional): A tuple/list of 2
                integers, specifying the height and width of the 2D kernel
                window. In case it's a single integer, it's value is used
                for all spatial dimensions. Default: 3.
            strides (int, tuple, list, optional): A tuple/list of 2
                integers, specifying the strides along the height and width
                of the 2D input. In case it's a single integer, it's value
                is used for all spatial dimensions. Default: 1.
            projection_ratio (int, optional): A scale factor applied to
                the number of input channels. The output of the first
                convolution will have ``input_channels // projection_ratio``.
                The goal is to decrease the number of parameters in the
                transposed convolution layer. Default: 4.
            padding (str, optional): One of "valid" or "same" (case-insensitive).
                Default: "same".
            output_shape: A tuple of integers specifying the shape of the output
                without the batch size. Default: None.
            bias (bool, optional): If ``True``, adds a learnable bias.
                Default: ``False``.
            name (string, optional): A string to identify this block.
                Default: Empty string.

        Returns:
            The output tensor of the block.

        """
        internal_filters = int_shape(input)[-1] // projection_ratio
        x = Conv2D(
            internal_filters,
            kernel_size=1,
            strides=1,
            padding=padding,
            use_bias=bias,
            name=name + '/conv2d_1'
        )(input)
        x = BatchNormalization(name=name + '/bn_1')(x)
        x = Activation('relu', name=name + '/relu_1')(x)

        # The shape of the following trasposed convolution is the output
        # shape of the block with 'internal_filters' channels
        x = Conv2DTranspose(
            internal_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=bias,
            name=name + '/transposed2d_1'
        )(x)
        x = BatchNormalization(name=name + '/bn_2')(x)
        x = Activation('relu', name=name + '/relu_2')(x)

        x = Conv2D(
            out_filters,
            kernel_size=1,
            strides=1,
            padding=padding,
            use_bias=bias,
            name=name + '/conv2d_2'
        )(x)
        x = BatchNormalization(name=name + '/bn_3')(x)
        x = Activation('relu', name=name + '/relu_3')(x)

        return x

    def dense_block(self, x, blocks, name):
        """A dense block.

        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        for i in range(blocks):
            x = self.conv_block(x, 8, name=name + '_block' + str(i + 1))
        return x

    @staticmethod
    def transition_block(x, reduction, name):
        """A transition block.

        # Arguments
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                               name=name + '_bn')(x)
        x = Activation('relu', name=name + '_relu')(x)
        x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
                   name=name + '_conv')(x)
        x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x

    @staticmethod
    def conv_block(x, growth_rate, name):
        """A building block for a dense block.

        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                name=name + '_0_bn')(x)
        x1 = Activation('relu', name=name + '_0_relu')(x1)
        x1 = Conv2D(4 * growth_rate, 1, use_bias=False,
                    name=name + '_1_conv')(x1)
        x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                name=name + '_1_bn')(x1)
        x1 = Activation('relu', name=name + '_1_relu')(x1)
        x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False,
                    name=name + '_2_conv')(x1)
        x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
        return x


def build_model(input_shape, use_shortcuts=True):
    ret = LinkNet(num_classes=1,
                  input_tensor=None,
                  input_shape=input_shape,
                  bias=False,
                  name='linknet')
    return ret.get_model(pretrained_encoder=False, use_shortcuts=use_shortcuts)


def get_network_name():
    return "dense_linknet"


if __name__ == '__main__':
    model = build_model(input_shape=(1024, 1024, 1))
    model.summary()
