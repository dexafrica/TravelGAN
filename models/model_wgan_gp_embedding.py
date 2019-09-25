import pandas as pd
import keras
# import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import tensorflow as tf

from keras import backend as K
from keras.layers import Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LSTM, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate, _Merge
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
from keras import backend as K

# import utils
from itertools import islice
from pyproj import Proj, transform
import s2sphere as s2
from dummyPy import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from functools import partial


class ModelWGANGP():
      
    def __init__(self):
        self.data_shape = (25,)
        self.noise_shape = (100,)
        
        # set hyper parameters for learning
        # optimizer = Adam(lr=5e-3, beta_1=0.5)
        self.n_critic = 5
        optimizer = RMSprop(lr=5e-5)

        # build the generator and the critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # -------------------------------------------
        # Computational Graph for the critic
        # -------------------------------------------

        # Freeze generator layer when training critic
        self.generator.trainable = False

        # Data input (real sample)
        real_data = Input(shape = self.data_shape)

        # Noise input
        z_disc = Input(shape=self.noise_shape)
        # Generate data based on noise(fake sample)
        fake_data = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_data)
        valid = self.critic(real_data)

        # Construct weighted average between real and fake data
        interpolated_data = RandomWeightedAverage() ([real_data, fake_data])
        # Determine validity of weighted sample
        validity_interpolated = self.critic( interpolated_data )

        # Loss function with additional averaged samples argument
        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples = interpolated_data)

        partial_gp_loss.__name__ = 'gradient_penalty'

        self.critic_model = Model(inputs=[real_data, z_disc],
                            outputs=[valid, fake, validity_interpolated])

        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                optimizer = optimizer,
                                loss_weights=[1, 1, 10])

        # -------------------------------------------
        # Computational Graph for the Generator
        # -------------------------------------------

        # Freeze the critic when training generator
        self.critic.trainable = False
        self.generator.trainable = True

        # sample noise for input to generator
        z_gen = Input(shape= self.noise_shape)
        # Geneate data based of noise
        data = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(data)
        # Defines the generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def save_model(self, version=None):
        if version is None:
            self.critic.save('wgan-gp/models/toy-mix-discriminator.h5')
            self.generator.save('wgan-gp/models/toy-mix-generator.h5')
            # self.combined.save('wgan-gp/models/toy-mix-combined.h5')
        else:
            self.critic.save('wgan-gp/models/toy-mix-discriminator-{}.h5'.format(version))
            self.generator.save('wgan-gp/models/toy-mix-generator-{}.h5'.format(version))
            # self.combined.save('wgan-gp/models/toy-mix-combined-{}.h5'.format(version))

      
    def load_model(self, version=None):
        if version is None:
            self.discriminator = load_model('wgan-gp/models/toy-mix-discriminator.h5')
            self.generator = load_model('wgan-gp/models/toy-mix-generator.h5')
            # self.combined = load_model('wgan-gp/models/toy-mix-combined.h5')
        else:
            self.discriminator = load_model('wgan-gp/models/toy-mix-discriminator-{}.h5'.format(version))
            self.generator = load_model('wgan-gp/models/toy-mix-generator-{}.h5'.format(version))
            # self.combined = load_model('wgan-gp/models/toy-mix-combined-{}.h5'.format(version))


    def build_generator(self):
        # BatchNormalization maintains the mean activation close to 0
        # and the activation standard deviation close to 1
        # noise = Input(shape=(self.noise_shape,))
        noise = Input(shape=self.noise_shape)
        
        hidden_1 = Dense(18)(noise)
        hidden_1 = LeakyReLU(alpha=0.2)(hidden_1)
        hidden_1 = BatchNormalization(momentum=0.8)(hidden_1)
        
        hidden_2 = Dense(16)(hidden_1)
        hidden_2 = LeakyReLU(alpha=0.2)(hidden_2)
        hidden_2 = BatchNormalization(momentum=0.8)(hidden_2)

         # numerical data [1,] - P_GRAGE
        branch_1_hidden_1 = Dense(100) (hidden_2)
        branch_1_hidden_1 = LeakyReLU(alpha=0.2) (branch_1_hidden_1)
        branch_1_hidden_1 = BatchNormalization(momentum=0.8) (branch_1_hidden_1)
        branch_1_hidden_2 = Dense(50)(branch_1_hidden_1)
        branch_1_hidden_2 = LeakyReLU(alpha=0.2) (branch_1_hidden_2)
        branch_1_hidden_2 = BatchNormalization(momentum=0.8) (branch_1_hidden_2)
        branch_1_output = Dense(1,  activation='sigmoid') (branch_1_hidden_2)

        # category data [1,] - P_AGE
        branch_2_hidden_1 = Dense(100) (hidden_2)
        branch_2_hidden_1 = LeakyReLU(alpha=0.2) (branch_2_hidden_1)
        branch_2_hidden_1 = BatchNormalization(momentum=0.8) (branch_2_hidden_1)
        branch_2_hidden_2 = Dense(50)(branch_2_hidden_1)
        branch_2_hidden_2 = LeakyReLU(alpha=0.2) (branch_2_hidden_2)
        branch_2_hidden_2 = BatchNormalization(momentum=0.8) (branch_2_hidden_2)
        branch_2_output = Dense(1,  activation='sigmoid') (branch_2_hidden_2)

        # categorical data [2,] - P_SEXE
        branch_3_hidden_1 = Dense(200)(hidden_2)
        branch_3_hidden_1 = LeakyReLU(alpha=0.2)(branch_3_hidden_1)
        branch_3_hidden_1 = BatchNormalization(momentum=0.8)(branch_3_hidden_1)
        branch_3_hidden_2 = Dense(100)(branch_3_hidden_1)
        branch_3_hidden_2 = LeakyReLU(alpha=0.2)(branch_3_hidden_2)
        branch_3_hidden_2 = BatchNormalization(momentum=0.8)(branch_3_hidden_2)
        branch_3_hidden_3 = Dense(50)(branch_3_hidden_2)
        branch_3_hidden_3 = LeakyReLU(alpha=0.2)(branch_3_hidden_3)
        branch_3_hidden_3 = BatchNormalization(momentum=0.8)(branch_3_hidden_3)
        branch_3_output = Dense(2, activation='softmax')(branch_3_hidden_3)

        
        # categorical data [8,] - P_STATUT
        branch_4_hidden_1 = Dense(200)(hidden_2)
        branch_4_hidden_1 = LeakyReLU(alpha=0.2)(branch_4_hidden_1)
        branch_4_hidden_1 = BatchNormalization(momentum=0.8)(branch_4_hidden_1)
        branch_4_hidden_2 = Dense(100)(branch_4_hidden_1)
        branch_4_hidden_2 = LeakyReLU(alpha=0.2)(branch_4_hidden_2)
        branch_4_hidden_2 = BatchNormalization(momentum=0.8)(branch_4_hidden_2)
        branch_4_hidden_3 = Dense(50)(branch_4_hidden_2)
        branch_4_hidden_3 = LeakyReLU(alpha=0.2)(branch_4_hidden_3)
        branch_4_hidden_3 = BatchNormalization(momentum=0.8)(branch_4_hidden_3)
        branch_4_output = Dense(8, activation='softmax')(branch_4_hidden_3)
        
        
        # categorical data [3,] - P_MOBIL
        branch_5_hidden_1 = Dense(200) (hidden_2)
        branch_5_hidden_1 = LeakyReLU(alpha=0.2) (branch_5_hidden_1)
        branch_5_hidden_1 = BatchNormalization(momentum=0.8) (branch_5_hidden_1)
        branch_5_hidden_2 = Dense(100)(branch_5_hidden_1)
        branch_5_hidden_2 = LeakyReLU(alpha=0.2)(branch_5_hidden_2)
        branch_5_hidden_2 = BatchNormalization(momentum=0.8)(branch_5_hidden_2)
        branch_5_hidden_3 = Dense(50)(branch_5_hidden_2)
        branch_5_hidden_3 = LeakyReLU(alpha=0.2)(branch_5_hidden_3)
        branch_5_hidden_3 = BatchNormalization(momentum=0.8)(branch_5_hidden_3)
        branch_5_output = Dense(3, activation='softmax')(branch_5_hidden_3)


        # categorical data [5,] - P_ORIG
        branch_6_hidden_1 = Dense(200) (hidden_2)
        branch_6_hidden_1 = LeakyReLU(alpha=0.2) (branch_6_hidden_1)
        branch_6_hidden_1 = BatchNormalization(momentum=0.8) (branch_6_hidden_1)
        branch_6_hidden_2 = Dense(100)(branch_6_hidden_1)
        branch_6_hidden_2 = LeakyReLU(alpha=0.2)(branch_6_hidden_2)
        branch_6_hidden_2 = BatchNormalization(momentum=0.8)(branch_6_hidden_2)
        branch_6_hidden_3 = Dense(50)(branch_6_hidden_2)
        branch_6_hidden_3 = LeakyReLU(alpha=0.2)(branch_6_hidden_3)
        branch_6_hidden_3 = BatchNormalization(momentum=0.8)(branch_6_hidden_3)
        branch_6_output = Dense(5, activation='sigmoid')(branch_6_hidden_3)


        # categorical data [5,] - P_DEST
        branch_7_hidden_1 = Dense(200) (hidden_2)
        branch_7_hidden_1 = LeakyReLU(alpha=0.2) (branch_7_hidden_1)
        branch_7_hidden_1 = BatchNormalization(momentum=0.8) (branch_7_hidden_1)
        branch_7_hidden_2 = Dense(100)(branch_7_hidden_1)
        branch_7_hidden_2 = LeakyReLU(alpha=0.2)(branch_7_hidden_2)
        branch_7_hidden_2 = BatchNormalization(momentum=0.8)(branch_7_hidden_2)
        branch_7_hidden_3 = Dense(50)(branch_7_hidden_2)
        branch_7_hidden_3 = LeakyReLU(alpha=0.2)(branch_7_hidden_3)
        branch_7_hidden_3 = BatchNormalization(momentum=0.8)(branch_7_hidden_3)
        branch_7_output = Dense(5, activation='sigmoid')(branch_7_hidden_3)

        # merge all datatypes output
        merged_output = concatenate([branch_1_output, branch_2_output, branch_3_output, branch_4_output, 
                                    branch_5_output, branch_6_output, branch_7_output])

        # branch_merged = Dense(100) (merged_output)
        # branch_merged = LeakyReLU(alpha=0.2)(branch_merged)
        # branch_merged = BatchNormalization(momentum=0.8)(branch_merged)
        # merged_output = Dense(25, activation='sigmoid')
        # model.add(Dense(1, activation='sigmoid'))
        return Model(inputs=noise, outputs=merged_output)


    def build_critic (self):
        model = Sequential()
        model.add(Dense(12, input_shape=self.data_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(6))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(1))
        model.summary()
        
        # Discriminator takes real data as an input and outputs its validity
        data = Input(shape=self.data_shape)
        validity = model(data)
        
        return Model(data, validity)

    def train(self, train_data, epochs=50001, batch_size=256, save_model_interval=1000):
        # Rescale train data to -1 to 1
        # train_data = ( train_data.astype(np.float32) - 127.5 ) / 127.5
        # train_data = np.expand_dims( train_data, axis=3 )

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):
                # ----------------------------
                # Train Discriminator
                # ----------------------------

                # Select a random batch of data
                idx = np.random.randint(0, train_data.shape[0], batch_size)
                data = train_data[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.noise_shape[0]))
                # Train the critic
                d_loss = self.critic_model.train_on_batch( [data, noise], [valid, fake, dummy] )


            # ---------------------------
            # Train Generator
            # ---------------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # Print progress
            # print ("{:5d} [D loss: {}, acc_real: {:2f}, acc_fake: {:2f}] [G loss: {}]".format(epoch, d_loss[0], 100*d_loss_real[1], 100*d_loss_fake[1], g_loss))

            with open('wgan-gp/logs/toy-mix-wgan-gp.log', 'a') as log_file:
                log_file.write('{},{}\n'.format(d_loss[0], g_loss))

            # If at save interval => save generated data samples
            if epoch != 0 and epoch % save_model_interval == 0:
                self.save_model(version=str(epoch))


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])



class EmbeddingMapping():
    """
    Helper class for handling categorical variables
    
    An instance of this class should be defined for each categorical variable we want to use.
    """
    def __init__(self, series):
        # get a list of unique values
        values = series.unique().tolist()
        
        # Set a dictionary mapping from values to integer value
        # In our example this will be {'Mercaz': 1, 'Old North': 2, 'Florentine': 3}
        self.embedding_dict = {value: int_value+1 for int_value, value in enumerate(values)}
        
        # The num_values will be used as the input_dim when defining the embedding layer. 
        # It will also be returned for unseen values 
        self.num_values = len(values) + 1

    def get_mapping(self, value):
        # If the value was seen in the training set, return its integer mapping
        if value in self.embedding_dict:
            return self.embedding_dict[value]
        
        # Else, return the same integer for unseen values
        else:
            return self.num_values

    def get_value(self, key):
        m = [x for x in self.embedding_dict if self.embedding_dict[x] == key]
        return m


class EmbeddingNetwork():
    
    def __init__(self):
        self.orig_size = 0
        self.dest_size = 0
        pass

    def rmse(self, y_true, y_pred):
        score = K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))
        return score

    def build_network(self):
      
        inputs = []
        embeddings = []

        input_orig_cat = Input(shape=(1,))
        embedding = Embedding(self.orig_size, 5, input_length=1)(input_orig_cat)
        output_orig = Reshape(target_shape=(5,))(embedding)
        inputs.append(input_orig_cat)
        embeddings.append(output_orig)
        
        input_dest_cat = Input(shape=(1,))
        embedding = Embedding(self.dest_size, 5, input_length=1)(input_dest_cat)
        output_dest = Reshape(target_shape=(5,))(embedding)
        inputs.append(input_dest_cat)
        embeddings.append(output_dest)
        
        output = concatenate([output_orig, output_dest])
        
        model = Model(inputs, output)
        model.compile(Adam(5e-4), loss='mse', metrics=[self.rmse])

        return model