import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from Painter import Painter
from Discriminator import Discriminator
from Generator import Generate
import tensorflow.keras.backend as K


if __name__ == "__main__":

    K.clear_session()

    input_shape = (1024,1024, 3)
    output_shape = (256, 256, 3)

    Dis = Discriminator(input_shape=output_shape)
    Dis.input()
    Dis.conv_pool_block(n_filter=16, filter_size=(5,5))
    Dis.conv_pool_block(n_filter=8, filter_size=(7,7))
    Dis.fully_connected(4)
    Dis.fully_connected(1) # for W-Loss
    Dis.build()

    Gen = Painter(Dis, input_shape, output_shape)
    Gen.input()
    Gen.conv_block(16, filter_size=(5,5), padding="same")
    Gen.conv_block(16, filter_size=(7,7), padding="same")
    Gen.pooling_block((2,2))
    Gen.conv_block(32, filter_size=(9,9), padding="same")
    Gen.conv_block(32, filter_size=(11,11), padding="same")
    Gen.pooling_block((2,2))
    Gen.top_block()
    #Gen.build_monitored([2,5],[0.5,0.5])
    Gen.build()



    painter_optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    disc_optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    for i in range(10):
        print("Training Discriminator...")
        for dis_training_loop in range(25):
            discri_grads = Dis.get_gradient()
            disc_optimizer.apply_gradients(zip(discri_grads, Dis.model.trainable_weights))
            print('\tDiscriminator loss: ',Dis.__current_loss__)
        

        print("Training Generator...")
        for gen_training_loop in range(10):
            painter_grads = Gen.get_gradient()

            print('\t\t\t'); _grads_summ=0
            for i in range(10):
                _grads_summ += painter_grads[i].numpy().sum()
            print("\tGenerator gradients sum: ",_grads_summ)

            painter_optimizer.apply_gradients(zip(painter_grads, Gen.model.trainable_weights))
            print('\tGenerator loss: ',Gen.__current_loss__)
        
        print(f"Loss at step {i}: Painter: {Gen.__current_loss__}; Discriminator: {Dis.__current_loss__}.")
