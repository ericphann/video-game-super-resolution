import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

def build_vgg_feature_extractor(layer_name='block5_conv4', input_shape=(None, None, 3)):
    """
    Build a truncated VGG19 model that outputs feature maps from a specific layer.
    """
    vgg = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    vgg.trainable = False
    output = vgg.get_layer(layer_name).output
    model = Model(inputs=vgg.input, outputs=output)
    model.trainable = False
    return model

def compute_perceptual_loss(vgg_model, hr_images, sr_images):
    """
    Compute L2 loss between VGG feature maps of real and super-res images.
    Inputs must be in [0, 1] range before preprocessing.
    """
    hr_processed = preprocess_input(hr_images * 255.0)
    sr_processed = preprocess_input(sr_images * 255.0)

    hr_features = vgg_model(hr_processed)
    sr_features = vgg_model(sr_processed)

    loss = tf.reduce_mean(tf.square(hr_features - sr_features))
    return loss
