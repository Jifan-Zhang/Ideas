import tensorflow as tf
from tensorflow.keras import Model
from Model_builder import Model_builder
from Generator import Generate
import numpy as np

class Painter(Model_builder):
    def __init__(self, input_shape = (1024,1024,3), output_shape = (512,512,3)):
        """
        The generator takes an input shape, output shape, and an instance of discriminator model.
        """
        super(Painter, self).__init__(input_shape, output_shape)

    def set_discriminator(self, Dis):
        self.discriminator = Dis

    def build_monitored(self, monitor, lbd):
        self.monitored = True
        """
        Create model monitoring some layer outputs. If accompanied with Style Loss
            monitor: list of unique positive integers. Indexing on which intermediate layer outputs to monitor the style (except the last layer).
        """
        if(len(monitor) != len(lbd)):
            raise RuntimeError("The lengths of monitor and lambda are not the same.")
        if("model" in self.__dict__):
            raise RuntimeError("Model for this painter has already been defined.")

        # The last layer loss accounts for highest (0.5) loss weight.
        lbd = [0.5*x for x in lbd] + [0.5]
        self.lbd = lbd
        outs=[]
        for i in range(len(self.blocks)):
            if i in monitor:
                outs.append(self.blocks[i])
                print(f"Monitoring -> {self.blocks[i].name}.")

        outs.append(self.blocks[-1])
        self.model = Model(inputs=self.blocks[0],outputs=outs)
        return self.model

    def build(self):
        """
        Create model without monitoring layer outputs.
        """
        self.monitored = False
        if("model" in self.__dict__):
            raise RuntimeError("Model for this painter has already been defined.")
        self.model = Model(inputs=self.blocks[0],outputs=self.blocks[-1])
        return self.model

    def __monitored_loss__(self, y_true, y_pred):
        loss = tf.Variable(0, dtype=tf.float32)
        """
        Implement Style Loss. Method returns a tf scalar.
        """
        count = 0
        for pred,true in zip(y_pred,y_true):
            n_sample = pred.shape[0]
            width = pred.shape[1]
            height = pred.shape[2]
            n_channel = pred.shape[3]
            pred_gram = tf.transpose(pred,[0,3,2,1])
            pred_gram = tf.reshape(pred_gram, (n_sample,n_channel,width*height)) # Flatten last 2 dims for matmul
            pred_gram = tf.matmul(pred_gram, tf.transpose(pred_gram, [0,2,1]))
            
            true_gram = tf.transpose(true,[0,3,2,1])
            true_gram = tf.reshape(true_gram, (n_sample,n_channel,width*height))
            true_gram = tf.matmul(true_gram, tf.transpose(true_gram, [0,2,1]))
            
            layer_loss = tf.reduce_sum(tf.math.square(pred_gram-true_gram))/(2*width*height)**2
            loss = loss + layer_loss*self.lbd[count]
            count += 1
        """
        Implement Discriminator loss
        """
        loss = loss + tf.cast(self.__loss__(y_pred[-1]), tf.float32)
        return loss

    def __loss__(self, y_pred):
        '''
        Implements W-loss
        '''
        score = self.discriminator.model(y_pred)
        loss = -score
        return loss

    def __get_pred__(self, path="color-gradient.jpg"):
        if(self.monitored):
            new_img = next(Generate(path, self.input_shape))
            y_true = self.model(new_img)
        else:
            y_true = None
        img_input = tf.random.uniform(minval=0, maxval=1, shape=(1,)+(self.input_shape),dtype=tf.float32)
        y_pred = self.model(img_input)
        #self.pred = y_pred
        return (y_true,y_pred)

    def get_gradient(self):
        with tf.GradientTape() as tape:
            if(self.monitored):
                loss = self.__monitored_loss__(*self.__get_pred__())
            else:
                loss = self.__loss__(self.__get_pred__()[1])
            self.__current_loss__ = loss.numpy()[0]
        return tape.gradient(loss, self.model.trainable_weights)

    def draw(self, *args):
        """
        This method generates new pictures after training. 
            Takes a input image seed or nothing.
        """
        if(len(args)==0):
            x = tf.random.uniform(minval=0, maxval=1, shape=(1,)+(self.input_shape),dtype=tf.float32)
            out = self.model(x)
            return out.numpy().astype(np.uint8)
        elif(len(args)>1):
            raise RuntimeError("draw method takes one argument.")
        else:
            x = args[0]
            input_shape = np.array(x).shape
            if(input_shape != self.input_shape):
                raise RuntimeError(f"Input shape should be {self.input_shape}, but received {input_shape}")
            out = self.model(tf.expand_dims(x,axis=0))
            return out.numpy().astype(np.uint8)
