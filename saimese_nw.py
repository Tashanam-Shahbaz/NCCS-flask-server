from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,  Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras import backend as K


def euclidean_distance(vects):
    '''Compute Euclidean Distance between two vectors'''
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 0.5
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def create_base_network(input_shape):
    '''Base Siamese Network'''
    
    seq = Sequential()
    seq.add(Conv2D(6, kernel_size=(5, 5), activation='relu', name='Conv1', strides=2, input_shape= input_shape, 
                        kernel_initializer='glorot_normal'))
    seq.add(MaxPooling2D((3,3), strides=(2, 2))) 
    seq.add(Dropout(.25))
    
    seq.add(Conv2D(12, kernel_size=(5, 5), activation='relu', name='Conv2', strides=2, kernel_initializer='glorot_normal'))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(.25))

    seq.add(Flatten(name='Flatten'))
    seq.add(Dense(128, activation='relu', kernel_initializer='glorot_normal')) 
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    
    return seq

def siamese_model_func():
    input_shape=(160, 160, 1)

    # network definition
    base_network = create_base_network(input_shape)

    input_a = Input(shape=(input_shape))
    input_b = Input(shape=(input_shape))

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Compute the Euclidean distance between the two vectors in the latent space
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(inputs=[input_a, input_b], outputs=distance)

    return model