import tensorflow as tf
from tensorflow.keras import Model
from Model_builder import Model_builder
from Generator import Generate

class Discriminator(Model_builder):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__(input_shape, (1,))

    def set_painter(self, Gen):
        self.Gen = Gen

    def build(self):
        """
        Create model
        """
        self.model = Model(inputs=self.blocks[0],outputs=self.blocks[-1])
        return self.model

    def __loss__(self, real_out, fake_out):
        return  fake_out - real_out

    def __get_pred__(self, path="color-gradient.jpg"):
        # fake inputs
        x = tf.random.uniform(minval=0, maxval=1, shape=(1,)+self.Gen.input_shape, dtype=tf.float32)
        if(self.Gen.monitored):
            fake_inputs = self.Gen.model(x)[-1]
        else:
            fake_inputs = self.Gen.model(x)
        fake_out = self.model(fake_inputs)

        # real inputs
        real_inputs = next(Generate("color-gradient.jpg", self.Gen.output_shape))
        real_out = self.model(real_inputs)
        return (real_out, fake_out)


    def get_gradient(self):
        with tf.GradientTape() as tape:
            real_out, fake_out = self.__get_pred__()
            loss = self.__loss__(real_out, fake_out)
        self.__current_loss__ = loss.numpy()[0]
        return tape.gradient(loss, self.model.trainable_weights)
