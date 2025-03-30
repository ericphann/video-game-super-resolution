import tensorflow as tf
import keras
from keras import initializers
import math
from keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
import datetime
import sys
import traceback
from keras.applications.vgg19 import VGG19, preprocess_input
tf.config.run_functions_eagerly(True)

# For Optuna hyperparameter tuning
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Set up multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Set mixed precision for memory efficiency, but be careful with dtype consistency
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU detected, enabling mixed precision...")
    # Only use mixed precision for specific operations
    # Keep PSNR calculations and VGG feature extraction in float32
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled. Using float16 for most operations and float32 for critical calculations.")
else:
    print("No GPU detected, using default precision.")

# Get dataset path from environment variable
dataset_path = os.environ.get('DATASET_PATH', './video-game-sr')
print(f"Using dataset path: {dataset_path}")

# Initial default values
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 100
NUM_RESIDUAL_BLOCKS = 16
LEAKYRELU_ALPHA = 0.2

# Set input shapes - from your notebook dimensions
LR_HEIGHT, LR_WIDTH = 270, 480
HR_HEIGHT, HR_WIDTH = 1080, 1920

# For memory efficiency, work with smaller patches from the images
PATCH_SCALE = 4  # Divide dimensions by this factor for memory efficiency
PATCH_LR_HEIGHT, PATCH_LR_WIDTH = LR_HEIGHT // PATCH_SCALE, LR_WIDTH // PATCH_SCALE  # 67, 120

# Calculate the actual HR dimensions after upscaling by 4x
PATCH_HR_HEIGHT, PATCH_HR_WIDTH = PATCH_LR_HEIGHT * 4, PATCH_LR_WIDTH * 4  # 268, 480

SCALE_FACTOR = 4  # 1080/270 = 4

# Directory for saving models and figures
OUTPUT_DIR = '/users/smicha17/optuna_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)

# Create a residual block with improved structure
def residual_block(x, filters, leakyrelu_alpha=0.2):
    # Initialize weights with He/Kaiming initialization for LeakyReLU
    scale_factor = math.sqrt(2.0 / (1 + leakyrelu_alpha**2))
    he_init = initializers.VarianceScaling(
        scale=scale_factor, 
        mode='fan_in', 
        distribution='normal'
    )
    
    # Store input for residual connection
    input_tensor = x
    
    # First convolution layer
    x = layers.Conv2D(filters, kernel_size=3, padding='same', 
                     kernel_initializer=he_init, 
                     bias_initializer='zeros')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=leakyrelu_alpha)(x)
    
    # Second convolution layer
    x = layers.Conv2D(filters, kernel_size=3, padding='same', 
                     kernel_initializer=he_init, 
                     bias_initializer='zeros')(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection (no activation before addition)
    x = layers.Add()([input_tensor, x])
    
    # Activation after skip connection
    x = layers.LeakyReLU(negative_slope=leakyrelu_alpha)(x)
    
    return x

# Create the generator model
def build_generator(leakyrelu_alpha=0.2, num_residual_blocks=16, filter_size=64):
    # Initialize weights with He/Kaiming initialization for LeakyReLU
    scale_factor = math.sqrt(2.0 / (1 + leakyrelu_alpha**2))
    he_init = initializers.VarianceScaling(
        scale=scale_factor, 
        mode='fan_in', 
        distribution='normal'
    )
    
    # Input is low-resolution image patch
    inputs = layers.Input(shape=(PATCH_LR_HEIGHT, PATCH_LR_WIDTH, 3))

    # Initial convolutional layer
    x = layers.Conv2D(filter_size, kernel_size=9, padding='same', 
                      kernel_initializer=he_init, 
                      bias_initializer='zeros')(inputs)
    x = layers.LeakyReLU(negative_slope=leakyrelu_alpha)(x)

    # Store the output of the first convolution for the residual connection
    skip_connection = x

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, filter_size, leakyrelu_alpha)

    # Post-residual convolutional layer
    x = layers.Conv2D(filter_size, kernel_size=3, padding='same', 
                     kernel_initializer=he_init, 
                     bias_initializer='zeros')(x)
    x = layers.BatchNormalization()(x)
    
    # Add the skip connection
    x = layers.Add()([x, skip_connection])

    # Upsampling blocks (4x upsampling for 270p -> 1080p)
    # More gradual filter reduction in upsampling path
    upscale_filters = [256, 128]
    for filters in upscale_filters:
        x = layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same', 
                                  kernel_initializer=he_init, 
                                  bias_initializer='zeros')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=leakyrelu_alpha)(x)

    # Final convolutional layer with more filters before RGB output
    x = layers.Conv2D(64, kernel_size=3, padding='same',
                     kernel_initializer=he_init,
                     bias_initializer='zeros')(x)
    x = layers.LeakyReLU(negative_slope=leakyrelu_alpha)(x)
    
    # Output layer
    outputs = layers.Conv2D(3, kernel_size=9, padding='same', activation='tanh', 
                           kernel_initializer=he_init, 
                           bias_initializer='zeros')(x)

    model = models.Model(inputs, outputs)
    return model

# Create a simpler discriminator model to work better in early training
def build_discriminator(leakyrelu_alpha=0.2, dropout_rate=0.2, filter_size=64):
    # Calculate initializer scale factor for Leaky ReLU
    scale_factor = math.sqrt(2.0 / (1 + leakyrelu_alpha**2))
    he_init = initializers.VarianceScaling(
        scale=scale_factor, 
        mode='fan_in', 
        distribution='normal'
    )
    
    # Input is high-resolution image patch
    inputs = layers.Input(shape=(PATCH_HR_HEIGHT, PATCH_HR_WIDTH, 3))

    # First conv layer
    x = layers.Conv2D(filter_size, kernel_size=4, strides=2, padding='same', 
                     kernel_initializer=he_init)(inputs)
    x = layers.LeakyReLU(negative_slope=leakyrelu_alpha)(x)
    
    # Simpler architecture with fewer layers
    filter_sizes = [filter_size*2, filter_size*4, filter_size*8]
    for filters in filter_sizes:
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding='same', 
                         kernel_initializer=he_init)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=leakyrelu_alpha)(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    x = layers.Flatten()(x)
    x = layers.Dense(1024, kernel_initializer=he_init)(x)
    x = layers.LeakyReLU(negative_slope=leakyrelu_alpha)(x)
    outputs = layers.Dense(1, activation='sigmoid', kernel_initializer=he_init)(x)

    return models.Model(inputs, outputs)

# Create VGG-based perceptual loss model
def build_vgg_feature_extractor():
    # Load pre-trained VGG19 model with ImageNet weights
    vgg = VGG19(weights='imagenet', include_top=False, input_shape=(PATCH_HR_HEIGHT, PATCH_HR_WIDTH, 3))
    
    # Freeze all VGG layers
    for layer in vgg.layers:
        layer.trainable = False
        
    # Use intermediate layer outputs for perceptual loss
    # Block 4, Conv 4 activation is commonly used for perceptual quality
    model = models.Model(inputs=vgg.input, outputs=vgg.get_layer('block4_conv4').output)
    
    # VGG models always expect float32 inputs
    tf.keras.backend.set_floatx('float32')
    return model
    
# Create adversarial model (Generator + Discriminator)
def build_adversarial_model(generator, discriminator):
    # Discriminator weights are frozen during generator training
    discriminator.trainable = False

    # GAN input is low-resolution image patch
    gan_input = layers.Input(shape=(PATCH_LR_HEIGHT, PATCH_LR_WIDTH, 3))

    # Generate high-resolution image
    gen_output = generator(gan_input)

    # Discriminator evaluates generator output
    gan_output = discriminator(gen_output)

    # Combined model (only generator weights are trainable)
    model = models.Model(gan_input, gan_output)
    return model

# Function to build and compile models with given hyperparameters
def build_and_compile_models(learning_rate, leakyrelu_alpha, num_residual_blocks, 
                            filter_size=64, dropout_rate=0.2, weight_decay=0):
    # Create optimizers with weight decay and gradient clipping
    if weight_decay > 0:
        optimizer_d = keras.optimizers.Adam(
            learning_rate=learning_rate,
            weight_decay=weight_decay,  # L2 regularization
            clipnorm=1.0  # Gradient clipping
        )
        optimizer_g = keras.optimizers.Adam(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=1.0
        )
    else:
        optimizer_d = keras.optimizers.Adam(
            learning_rate=learning_rate, 
            clipnorm=1.0
        )
        optimizer_g = keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0
        )
    
    # Create models with tuned hyperparameters
    generator = build_generator(
        leakyrelu_alpha=leakyrelu_alpha,
        num_residual_blocks=num_residual_blocks,
        filter_size=filter_size
    )

    discriminator = build_discriminator(
        leakyrelu_alpha=leakyrelu_alpha,
        dropout_rate=dropout_rate,
        filter_size=filter_size
    )

    adversarial_model = build_adversarial_model(generator, discriminator)

    # Compile discriminator
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=optimizer_d,
        metrics=['accuracy']
    )

    # Compile adversarial model
    adversarial_model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer_g
    )

    return generator, discriminator, adversarial_model

# Check if dataset structure exists
def check_dataset():
    dirs = [
        os.path.join(dataset_path, "training-lr"),
        os.path.join(dataset_path, "training-hr"),
        os.path.join(dataset_path, "validation-lr"),
        os.path.join(dataset_path, "validation-hr"),
        os.path.join(dataset_path, "test-lr"),
        os.path.join(dataset_path, "test-hr")
    ]
    
    all_exist = True
    for d in dirs:
        if not os.path.exists(d):
            print(f"WARNING: Directory {d} does not exist")
            all_exist = False
        else:
            files = os.listdir(d)
            num_images = len([f for f in files if f.endswith('.png') or f.endswith('.jpg')])
            print(f"Directory {d} exists with {num_images} image files")
            
    return all_exist

def create_sr_gan_dataset(lr_dir, hr_dir, batch_size=4, shuffle=True, buffer_size=1000):
    """
    Create a TensorFlow dataset for super-resolution GAN training.
    Handles 270p to 1080p image pairs.
    """
    print(f"Creating dataset from {lr_dir} and {hr_dir} with batch size {batch_size}")
    sys.stdout.flush()  # Force output to be displayed
    
    # Get sorted lists of file paths
    lr_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)
                      if f.endswith('.png') or f.endswith('.jpg')])
    hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)
                      if f.endswith('.png') or f.endswith('.jpg')])
                      
    print(f"Found {len(lr_files)} LR images and {len(hr_files)} HR images")
    
    if len(lr_files) == 0 or len(hr_files) == 0:
        print(f"WARNING: Empty dataset directories found!")
        print(f"LR directory: {lr_dir}, HR directory: {hr_dir}")
        for d in [lr_dir, hr_dir]:
            print(f"Directory {d} contains: {os.listdir(d)}")
        raise ValueError("Empty dataset directories")
    
    # Print out the first few file paths to verify correct structure
    print(f"First 3 LR images: {lr_files[:3]}")
    print(f"First 3 HR images: {hr_files[:3]}")
    sys.stdout.flush()

    # Create TensorFlow datasets from file paths
    lr_ds = tf.data.Dataset.from_tensor_slices(lr_files)
    hr_ds = tf.data.Dataset.from_tensor_slices(hr_files)
    
    print("Created TensorFlow tensor datasets, now defining preprocessing functions...")
    sys.stdout.flush()

    # Define preprocessing functions
    print("Defining preprocessing functions...")
    
    def preprocess_lr(image_path):
        try:
            img = tf.io.read_file(image_path)
            img = tf.image.decode_png(img, channels=3)
            
            # Convert to float32 FIRST
            img = tf.image.convert_image_dtype(img, tf.float32)
            
            # THEN resize to the expected patch size (to avoid overflow)
            img = tf.image.resize(img, [PATCH_LR_HEIGHT, PATCH_LR_WIDTH])
            
            # Normalize to [-1, 1]
            img = (img * 2) - 1
            
            # Extra safety check for normalization
            img = tf.clip_by_value(img, -1.0, 1.0)
            
            return img
        except Exception as e:
            print(f"Error preprocessing LR image {image_path}: {str(e)}")
            raise

    def preprocess_hr(image_path):
        try:
            img = tf.io.read_file(image_path)
            img = tf.image.decode_png(img, channels=3)
            
            # Convert to float32 FIRST
            img = tf.image.convert_image_dtype(img, tf.float32)
            
            # THEN resize to HR size (to avoid overflow)
            img = tf.image.resize(img, [PATCH_HR_HEIGHT, PATCH_HR_WIDTH])
            
            # Normalize to [-1, 1]
            img = (img * 2) - 1
            
            # Extra safety check for normalization
            img = tf.clip_by_value(img, -1.0, 1.0)
            
            return img
        except Exception as e:
            print(f"Error preprocessing HR image {image_path}: {str(e)}")
            raise
            
    print("Applying preprocessing functions...")
    sys.stdout.flush()

    # Load and preprocess the images
    try:
        print("Mapping preprocessing to LR dataset...")
        lr_ds = lr_ds.map(preprocess_lr, num_parallel_calls=tf.data.AUTOTUNE)
        
        print("Mapping preprocessing to HR dataset...")
        hr_ds = hr_ds.map(preprocess_hr, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Test first element of each dataset to verify preprocessing works
        print("Testing preprocessing on first image...")
        for lr_img in lr_ds.take(1):
            print(f"LR shape after preprocessing: {lr_img.shape}, dtype: {lr_img.dtype}")
            print(f"LR value range: [{tf.reduce_min(lr_img).numpy():.2f}, {tf.reduce_max(lr_img).numpy():.2f}]")
            
        for hr_img in hr_ds.take(1):
            print(f"HR shape after preprocessing: {hr_img.shape}, dtype: {hr_img.dtype}")
            print(f"HR value range: [{tf.reduce_min(hr_img).numpy():.2f}, {tf.reduce_max(hr_img).numpy():.2f}]")
        
        print("Preprocessing test successful")
        sys.stdout.flush()
    except Exception as e:
        print(f"Error during preprocessing map: {str(e)}")
        traceback.print_exc()
        raise

    # Zip the datasets together
    print("Zipping datasets together...")
    dataset = tf.data.Dataset.zip((lr_ds, hr_ds))
    
    # Test the zipped dataset
    print("Testing zipped dataset...")
    for lr_test, hr_test in dataset.take(1):
        print(f"Zipped dataset first element - LR: {lr_test.shape}, HR: {hr_test.shape}")
    sys.stdout.flush()

    # Shuffle with a good buffer size for GANs
    if shuffle:
        print(f"Shuffling dataset with buffer size {buffer_size}...")
        dataset = dataset.shuffle(buffer_size=buffer_size)
        print("Shuffling complete")

    # Use smaller batch sizes for GAN training
    print(f"Batching dataset with batch size {batch_size}...")
    dataset = dataset.batch(batch_size)
    print("Batching complete")
    
    # Prefetch for performance
    print("Applying prefetch...")
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    print("Dataset creation complete!")
    
    # Test the final dataset
    print("Testing final dataset...")
    for test_batch in dataset.take(1):
        test_lr, test_hr = test_batch
        print(f"First batch - LR: {test_lr.shape}, HR: {test_hr.shape}")
    sys.stdout.flush()

    return dataset

# Train dataset wrapper function to ensure the model and data dimensions match
def get_patch_dataset(dataset_path, batch_size=2, shuffle=True):
    """
    Create datasets with consistent dimensions that match the model's expectations
    """
    train_lr_dir = os.path.join(dataset_path, "training-lr")
    train_hr_dir = os.path.join(dataset_path, "training-hr")
    val_lr_dir = os.path.join(dataset_path, "validation-lr")
    val_hr_dir = os.path.join(dataset_path, "validation-hr")
    test_lr_dir = os.path.join(dataset_path, "test-lr")
    test_hr_dir = os.path.join(dataset_path, "test-hr")

    # Create datasets with patching for memory efficiency
    print(f"Creating training dataset from {train_lr_dir} and {train_hr_dir}...")
    train_ds = create_sr_gan_dataset(
        train_lr_dir, train_hr_dir,
        batch_size=batch_size,
        shuffle=shuffle
    )
    print(f"Training dataset created successfully with batch size {batch_size}")

    print(f"Creating validation dataset from {val_lr_dir} and {val_hr_dir}...")
    val_ds = create_sr_gan_dataset(
        val_lr_dir, val_hr_dir,
        batch_size=batch_size,
        shuffle=False
    )
    print("Validation dataset created successfully")

    print(f"Creating test dataset from {test_lr_dir} and {test_hr_dir}...")
    test_ds = create_sr_gan_dataset(
        test_lr_dir, test_hr_dir,
        batch_size=batch_size,
        shuffle=False
    )
    print("Test dataset created successfully")

    return train_ds, val_ds, test_ds

# Training loop for GAN with perceptual loss
def train_gan(generator, discriminator, adversarial_model, train_ds, val_ds, 
             epochs=100, trial_number=None, log_interval=10, 
             d_learning_rate=None, g_learning_rate=None, 
             l1_weight=1.0, perceptual_weight=0.001, label_smoothing=0.0,
             use_relativistic=True, lr_decay_factor=0.5, lr_decay_epochs=30):
    # Create lists to store metrics
    d_losses = []
    g_losses = []
    psnr_history = []
    
    # Create optimizers - allow different learning rates for discriminator and generator
    if d_learning_rate is None:
        d_learning_rate = LEARNING_RATE
    if g_learning_rate is None:
        g_learning_rate = LEARNING_RATE
        
    # Create learning rate schedules
    d_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=d_learning_rate,
        decay_steps=lr_decay_epochs * (len(list(train_ds)) // 10),  # Decay every lr_decay_epochs
        decay_rate=lr_decay_factor,
        staircase=True
    )
    
    g_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=g_learning_rate,
        decay_steps=lr_decay_epochs * (len(list(train_ds)) // 10),
        decay_rate=lr_decay_factor,
        staircase=True
    )
    
    d_optimizer = keras.optimizers.Adam(
        learning_rate=d_lr_schedule,
        clipnorm=1.0  # Gradient clipping
    )
    
    g_optimizer = keras.optimizers.Adam(
        learning_rate=g_lr_schedule,
        clipnorm=1.0  # Gradient clipping
    )
    
    # Create perceptual loss model if needed
    if perceptual_weight > 0:
        vgg_model = build_vgg_feature_extractor()
    else:
        vgg_model = None
    
    # BCE loss for adversarial training
    bce_loss = keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    
    # L1 loss for content preservation
    mae_loss = keras.losses.MeanAbsoluteError()
    
    # MSE loss for relativistic GAN or perceptual loss
    mse_loss = keras.losses.MeanSquaredError()
    
    # Performance metric - PSNR
    psnr_metric = keras.metrics.Mean()
    
    # Convert [-1, 1] range to [0, 1] for VGG input
    def preprocess_vgg(images):
        # Convert from [-1, 1] to [0, 1]
        images = (images + 1) / 2.0
        # Convert to RGB and scale for VGG (values centered around ImageNet mean)
        images = tf.keras.applications.vgg19.preprocess_input(images * 255.0)
        return images
    
    # Convert datasets to distributed format 
    train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
    val_dist_ds = strategy.experimental_distribute_dataset(val_ds)
    
    # Define training step
    def train_step(lr_images, hr_images):
        print(f"  Processing batch - LR shape: {lr_images.shape}, HR shape: {hr_images.shape}")
        batch_size = tf.shape(lr_images)[0]
        
        # Labels for real/fake images
        # Apply label smoothing for real labels (e.g., 0.9 instead of 1.0)
        if label_smoothing > 0:
            real_labels = tf.ones((batch_size, 1)) * (1.0 - label_smoothing)
        else:
            real_labels = tf.ones((batch_size, 1))
            
        # Fake labels - always 0
        fake_labels = tf.zeros((batch_size, 1))
        
        # Train discriminator
        with tf.GradientTape() as d_tape:
            # Generate fake images
            fake_hr_images = generator(lr_images, training=True)
            
            # Get discriminator predictions
            real_predictions = discriminator(hr_images, training=True)
            fake_predictions = discriminator(fake_hr_images, training=True)
            
            # Calculate discriminator losses
            if use_relativistic:
                # Relativistic average discriminator loss
                d_loss_real = mse_loss(real_labels, 
                                      real_predictions - tf.reduce_mean(fake_predictions))
                d_loss_fake = mse_loss(fake_labels, 
                                      fake_predictions - tf.reduce_mean(real_predictions))
            else:
                # Standard GAN loss
                d_loss_real = bce_loss(real_labels, real_predictions)
                d_loss_fake = bce_loss(fake_labels, fake_predictions)
                
            d_loss = (d_loss_real + d_loss_fake) * 0.5
        
        # Debugging output for discriminator
        print(f"D loss components - Real: {d_loss_real:.4f}, Fake: {d_loss_fake:.4f}")
        print(f"Real preds range: [{tf.reduce_min(real_predictions):.4f}, {tf.reduce_max(real_predictions):.4f}]")
        print(f"Fake preds range: [{tf.reduce_min(fake_predictions):.4f}, {tf.reduce_max(fake_predictions):.4f}]")
        print("Computing discriminator gradients...")
            
        # Compute and apply discriminator gradients
        d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
        
        # Add debugging for gradients
        if d_gradients:
            print(f"Number of discriminator gradients: {len(d_gradients)}")
            # Check if all gradients are None
            all_none = all(g is None for g in d_gradients)
            print(f"All gradients None: {all_none}")
            # Check if any gradient is NaN
            has_nan = any(tf.reduce_any(tf.math.is_nan(g)) if g is not None else False for g in d_gradients)
            print(f"Has NaN gradients: {has_nan}")
        
        # Better gradient handling
        if d_gradients and any(g is not None for g in d_gradients):
            # Clip gradients to prevent explosion
            d_gradients, _ = tf.clip_by_global_norm(d_gradients, 1.0)
            
            # Filter out None gradients
            grads_and_vars = [(g, v) for g, v in zip(d_gradients, discriminator.trainable_variables) if g is not None]
            if grads_and_vars:
                d_optimizer.apply_gradients(grads_and_vars)
            else:
                print("Warning: All discriminator gradients are None")
        else:
            print("Warning: No valid discriminator gradients to apply")
        
        # Train generator
        with tf.GradientTape() as g_tape:
            # Generate fake images
            fake_hr_images = generator(lr_images, training=True)
            
            # Get discriminator predictions for the fake images
            fake_predictions = discriminator(fake_hr_images, training=False)
            real_predictions = discriminator(hr_images, training=False)
            
            # Adversarial loss (fool the discriminator)
            if use_relativistic:
                # Relativistic average generator loss (inverted labels)
                g_adversarial_loss = mse_loss(real_labels, 
                                            fake_predictions - tf.reduce_mean(real_predictions))
            else:
                # Standard GAN loss
                g_adversarial_loss = bce_loss(real_labels, fake_predictions)
            
            # Content loss (L1 loss)
            content_loss = mae_loss(hr_images, fake_hr_images) * l1_weight
            
            # Start with just L1 loss for first few epochs
            if epoch < 2:
                g_loss = content_loss
                print("Using L1 loss only for initial stabilization")
            else:
                # Combine with adversarial loss after model has stabilized
                g_loss = g_adversarial_loss + content_loss
            
            # Add perceptual (VGG) loss if enabled
            if perceptual_weight > 0 and vgg_model is not None and epoch >= 3:
                # Add perceptual loss only after model has stabilized further
                # Preprocess images for VGG - explicit casting to float32
                hr_vgg_input = tf.cast(preprocess_vgg(hr_images), tf.float32)
                fake_hr_vgg_input = tf.cast(preprocess_vgg(fake_hr_images), tf.float32)
                
                # Extract VGG features
                hr_features = vgg_model(hr_vgg_input)
                fake_hr_features = vgg_model(fake_hr_vgg_input)
                
                # Compute perceptual loss with consistent dtype
                perceptual_loss = mse_loss(hr_features, fake_hr_features) * tf.cast(perceptual_weight, tf.float32)
                g_loss += perceptual_loss
                print(f"Added perceptual loss: {perceptual_loss:.4f}")
            
        # Compute and apply generator gradients
        g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
        
        # Better gradient handling for generator
        if g_gradients and any(g is not None for g in g_gradients):
            # Clip gradients to prevent explosion
            g_gradients, _ = tf.clip_by_global_norm(g_gradients, 1.0)
            
            # Filter out None gradients
            grads_and_vars = [(g, v) for g, v in zip(g_gradients, generator.trainable_variables) if g is not None]
            if grads_and_vars:
                g_optimizer.apply_gradients(grads_and_vars)
            else:
                print("Warning: All generator gradients are None")
        else:
            print("Warning: No valid generator gradients to apply")
            
        return d_loss, g_loss
    
    # Define distributed training step
    @tf.function
    def distributed_train_step(lr_images, hr_images):
        per_replica_losses = strategy.run(train_step, args=(lr_images, hr_images))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    # Helper function to compute PSNR
    def compute_psnr(hr_image, sr_image):
        # Convert from [-1,1] to [0,1] if needed
        if tf.reduce_min(hr_image) < 0:
            hr_image = (hr_image + 1) / 2.0
            sr_image = (sr_image + 1) / 2.0
            
        # Compute MSE - ensure same dtype for all operations by casting to float32
        hr_image = tf.cast(hr_image, tf.float32)
        sr_image = tf.cast(sr_image, tf.float32)
        mse = tf.reduce_mean(tf.square(hr_image - sr_image))
        
        if mse < 1e-10:
            return tf.constant(100.0, dtype=tf.float32)  # Perfect match
        
        # Calculate PSNR - everything in float32
        max_pixel = tf.constant(1.0, dtype=tf.float32)
        psnr = 20.0 * tf.math.log(max_pixel / tf.sqrt(mse)) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
        return psnr

    # Training loop
    try:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            # Reset metrics for each epoch
            d_loss_epoch = []
            g_loss_epoch = []
            psnr_metric.reset_states()
            
            # Train on batches
            batch_count = 0
            for batch_idx, (lr_batch, hr_batch) in enumerate(train_dist_ds):
                try:
                    # Print batch info before training
                    print(f"Starting batch {batch_idx} of epoch {epoch+1}...")
                    start_time = time.time()  # Track batch processing time
                    
                    # Train on batch using distributed training
                    d_loss, g_loss = distributed_train_step(lr_batch, hr_batch)
                    
                    # Convert from tensor to numpy for logging
                    d_loss_np = d_loss.numpy()
                    g_loss_np = g_loss.numpy()
                    d_loss_epoch.append(d_loss_np)
                    g_loss_epoch.append(g_loss_np)
                    
                    # Calculate batch processing time
                    batch_time = time.time() - start_time
                    
                    # Print detailed progress for every batch
                    print(f"Batch {batch_idx}: D loss: {d_loss_np:.4f}, G loss: {g_loss_np:.4f} (took {batch_time:.2f}s)")
                    
                    # Print more detailed info at intervals
                    if batch_idx % log_interval == 0:
                        current_d_lr = d_optimizer.learning_rate(d_optimizer.iterations).numpy()
                        current_g_lr = g_optimizer.learning_rate(g_optimizer.iterations).numpy()
                        print(f"  Current learning rates - D: {current_d_lr:.2e}, G: {current_g_lr:.2e}")
                        print(f"  Processed {batch_idx+1}/{len(list(train_ds))} batches in epoch {epoch+1}")
                    
                    batch_count += 1
                except tf.errors.ResourceExhaustedError:
                    print(f"Warning: Out of memory in batch {batch_idx}, skipping")
                    # Try to free memory
                    import gc
                    gc.collect()
                    tf.keras.backend.clear_session()
                    continue
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue

            # Check if we processed any batches
            if batch_count == 0:
                print("No batches were processed in this epoch, stopping training")
                return [], [], 0  # Return empty lists and poor PSNR
                
            # Compute epoch average losses
            avg_d_loss = np.mean(d_loss_epoch)
            avg_g_loss = np.mean(g_loss_epoch)
            d_losses.append(avg_d_loss)
            g_losses.append(avg_g_loss)

            # Evaluate on validation set
            print("\nEvaluating on validation set...")
            val_psnr_values = []
            val_batch_count = 0
            
            try:
                for val_lr_batch, val_hr_batch in val_dist_ds.take(5):  # Evaluate on 5 batches
                    print(f"  Processing validation batch {val_batch_count+1}/5")
                    val_sr_batch = generator(val_lr_batch, training=False)
                    
                    # Compute PSNR for each image in the batch
                    batch_psnrs = []
                    for i in range(val_lr_batch.shape[0]):
                        psnr = compute_psnr(val_hr_batch[i], val_sr_batch[i])
                        psnr_val = psnr.numpy()
                        batch_psnrs.append(psnr_val)
                        val_psnr_values.append(psnr_val)
                    
                    print(f"  Validation batch {val_batch_count+1} - Mean PSNR: {np.mean(batch_psnrs):.2f} dB")
                    val_batch_count += 1
            except Exception as e:
                print(f"Error during validation: {str(e)}")
                traceback.print_exc()
            
            # Compute average PSNR
            avg_psnr = np.mean(val_psnr_values) if val_psnr_values else 0
            psnr_history.append(avg_psnr)
            
            # Get current learning rates
            current_d_lr = d_optimizer.learning_rate(d_optimizer.iterations).numpy()
            current_g_lr = g_optimizer.learning_rate(g_optimizer.iterations).numpy()
            
            print(f"Epoch {epoch+1} - D loss: {avg_d_loss:.4f}, G loss: {avg_g_loss:.4f}, PSNR: {avg_psnr:.2f} dB")
            print(f"Learning rates - D: {current_d_lr:.2e}, G: {current_g_lr:.2e}")
            
            # For Optuna trials, we need to report intermediate values
            if trial_number is not None:
                # Report to Optuna for pruning if needed
                pruned = optuna.report(avg_psnr, epoch)
                if pruned:
                    # Break the loop if Optuna suggests pruning
                    print("Trial pruned by Optuna")
                    raise optuna.TrialPruned()

            # Save a sample image during training 
            if (epoch + 1) % 5 == 0 or epoch == 0:
                for lr_batch, hr_batch in val_ds.take(1):
                    lr_image = lr_batch[0]
                    hr_image = hr_batch[0]

                    # Generate SR image
                    sr_image = generator(tf.expand_dims(lr_image, 0))[0]

                    # Convert images from [-1, 1] to [0, 1] for display
                    lr_image = (lr_image + 1) / 2.0
                    hr_image = (hr_image + 1) / 2.0
                    sr_image = (sr_image + 1) / 2.0

                    # Plot the images
                    plt.figure(figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.title('Low Resolution')
                    plt.imshow(lr_image)

                    plt.subplot(1, 3, 2)
                    plt.title(f'Super Resolution (PSNR: {compute_psnr(hr_image, sr_image):.2f}dB)')
                    plt.imshow(sr_image)

                    plt.subplot(1, 3, 3)
                    plt.title('High Resolution (Ground Truth)')
                    plt.imshow(hr_image)

                    # Save with trial number if provided
                    save_path = os.path.join(OUTPUT_DIR, 'figures')
                    if trial_number is not None:
                        save_path = os.path.join(save_path, f'trial_{trial_number}')
                        os.makedirs(save_path, exist_ok=True)
                    
                    plt.savefig(os.path.join(save_path, f'epoch_{epoch+1}_sample.png'))
                    plt.close()
                    
                    # Save the generator model checkpoint
                    if epoch > 0 and epoch % 10 == 0:
                        model_save_path = os.path.join(OUTPUT_DIR, 'models')
                        if trial_number is not None:
                            model_save_path = os.path.join(model_save_path, f'trial_{trial_number}')
                            os.makedirs(model_save_path, exist_ok=True)
                        
                        generator.save(os.path.join(model_save_path, f'generator_epoch_{epoch+1}.keras'))
                    
    except optuna.TrialPruned:
        print("Trial was pruned by Optuna")
        # Calculate final metrics before pruning
        final_psnr = psnr_history[-1] if psnr_history else 0
        return d_losses, g_losses, final_psnr
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Return best achieved PSNR if available
        final_psnr = max(psnr_history) if psnr_history else 0
        return d_losses, g_losses, final_psnr

    # Final PSNR evaluation on validation set
    final_psnr_values = []
    for val_lr_batch, val_hr_batch in val_ds.take(10):  # More comprehensive evaluation
        val_sr_batch = generator(val_lr_batch, training=False)
        
        # Compute PSNR for each image in the batch
        for i in range(val_lr_batch.shape[0]):
            psnr = compute_psnr(val_hr_batch[i], val_sr_batch[i])
            final_psnr_values.append(psnr.numpy())
    
    final_psnr = np.mean(final_psnr_values) if final_psnr_values else 0
    print(f"Final validation PSNR: {final_psnr:.2f} dB")

    return d_losses, g_losses, final_psnr

# Define objective function for Optuna
def objective(trial):
    # Get trial number for logging
    trial_number = trial.number
    
    # Generate unique directory for this trial
    trial_dir = os.path.join(OUTPUT_DIR, f'trial_{trial_number}')
    os.makedirs(trial_dir, exist_ok=True)
    
    # Define hyperparameters to optimize with narrower ranges
    learning_rate = trial.suggest_float('learning_rate', 5e-5, 5e-4, log=True)
    leakyrelu_alpha = trial.suggest_float('leakyrelu_alpha', 0.1, 0.3)
    num_residual_blocks = trial.suggest_int('num_residual_blocks', 8, 16, step=2)
    filter_size = trial.suggest_categorical('filter_size', [48, 64, 96])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [2, 4])
    
    # Advanced GAN training techniques - with reduced perceptual weight
    d_learning_rate = trial.suggest_float('d_learning_rate', learning_rate * 0.5, learning_rate * 1.5, log=True)
    l1_weight = trial.suggest_float('l1_weight', 0.1, 10.0, log=True)
    perceptual_weight = trial.suggest_float('perceptual_weight', 0.0001, 0.01, log=True)  # Reduced range
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.1)
    use_relativistic = trial.suggest_categorical('use_relativistic', [True, False])
    
    print(f"\nTrial {trial_number}: Testing parameters:")
    print(f"learning_rate: {learning_rate}, alpha: {leakyrelu_alpha}, blocks: {num_residual_blocks}")
    print(f"filter_size: {filter_size}, dropout: {dropout_rate}, weight_decay: {weight_decay}")
    print(f"batch_size: {batch_size}, d_lr: {d_learning_rate}, l1_weight: {l1_weight}")
    print(f"perceptual_weight: {perceptual_weight}, label_smoothing: {label_smoothing}")
    print(f"use_relativistic: {use_relativistic}")

    # Check dataset structure
    if not check_dataset():
        print("Dataset structure issue detected. Please check the dataset path and directory structure.")
        return 0.0  # Return poor score
        
    try:
        print("\n========== STARTING DATASET CREATION ==========")
        sys.stdout.flush()  # Force output to be displayed immediately
        
        # Create datasets with selected batch size
        train_ds, val_ds, test_ds = get_patch_dataset(dataset_path, batch_size=batch_size)
        
        # Add the validation code:
        print("Validating dataset normalization...")
        for lr_batch, hr_batch in train_ds.take(1):
            print(f"LR range: [{tf.reduce_min(lr_batch).numpy():.2f}, {tf.reduce_max(lr_batch).numpy():.2f}]")
            print(f"HR range: [{tf.reduce_min(hr_batch).numpy():.2f}, {tf.reduce_max(hr_batch).numpy():.2f}]")
            
            # If range is not approximately [-1, 1], you have a preprocessing issue
            if tf.reduce_max(lr_batch).numpy() > 2.0 or tf.reduce_min(lr_batch).numpy() < -2.0:
                print("WARNING: LR data is not properly normalized!")
                print("You should fix the preprocessing function in create_sr_gan_dataset")
            
            if tf.reduce_max(hr_batch).numpy() > 2.0 or tf.reduce_min(hr_batch).numpy() < -2.0:
                print("WARNING: HR data is not properly normalized!")
                print("You should fix the preprocessing function in create_sr_gan_dataset")

        print("\n========== DATASET CREATION COMPLETED ==========")
        print(f"Inspecting dataset sizes:")
        
        # Check dataset sizes and print first batch info
        try:
            print("Checking training dataset...")
            train_size = len(list(train_ds))
            print(f"Training dataset has {train_size} batches")
            
            # Try to extract one batch to verify dataset is working
            print("Extracting one training batch to verify...")
            for lr_batch, hr_batch in train_ds.take(1):
                print(f"Training batch shapes - LR: {lr_batch.shape}, HR: {hr_batch.shape}")
                print(f"Data types - LR: {lr_batch.dtype}, HR: {hr_batch.dtype}")
                print(f"Value ranges - LR: [{tf.reduce_min(lr_batch).numpy():.2f}, {tf.reduce_max(lr_batch).numpy():.2f}], "
                      f"HR: [{tf.reduce_min(hr_batch).numpy():.2f}, {tf.reduce_max(hr_batch).numpy():.2f}]")
                break
                
            print("Validation dataset inspection...")
            val_size = len(list(val_ds))
            print(f"Validation dataset has {val_size} batches")
        except Exception as e:
            print(f"Error inspecting datasets: {str(e)}")
            traceback.print_exc()
        
        sys.stdout.flush()  # Force output to be displayed
        print("\n========== BUILDING MODELS ==========")
        
        # Build models with trial hyperparameters under the strategy scope
        with strategy.scope():
            print(f"\nBuilding generator with {num_residual_blocks} residual blocks and {filter_size} filters...")
            print(f"Using leaky ReLU with negative_slope={leakyrelu_alpha}, dropout={dropout_rate}, weight_decay={weight_decay}")
            generator, discriminator, adversarial_model = build_and_compile_models(
                learning_rate=learning_rate,
                leakyrelu_alpha=leakyrelu_alpha,
                num_residual_blocks=num_residual_blocks,
                filter_size=filter_size,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay
            )
            
        print("Generator summary:")
        generator.summary(line_length=120, print_fn=print)
        print("\nSkipping discriminator summary to avoid potential hang")
        #print("\nDiscriminator summary:") -- skipping these two lines for now
        #discriminator.summary(line_length=120, print_fn=print)
        
        # Train for fewer epochs during optimization
        optimization_epochs = 5  # Reduced for faster trials
        
        # Train the model
        print(f"Training model for {optimization_epochs} epochs...")
        
        # Train for minimal epochs
        print("\n========== STARTING MINIMAL TRAINING ==========")
        
        # Convert datasets to distributed format
        train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
        val_dist_ds = strategy.experimental_distribute_dataset(val_ds)
        
        # Train the model with distributed strategy
        d_losses, g_losses, final_psnr = train_gan(
            generator, 
            discriminator, 
            adversarial_model,
            train_ds,  # Original dataset for reference
            val_ds,    # Original dataset for reference
            epochs=optimization_epochs,
            trial_number=trial_number,
            d_learning_rate=d_learning_rate,
            g_learning_rate=learning_rate,
            l1_weight=l1_weight,
            perceptual_weight=perceptual_weight,
            label_smoothing=label_smoothing,
            use_relativistic=use_relativistic
        )
        
        # Calculate average PSNR
        print(f"Trial {trial_number} achieved average PSNR: {final_psnr:.2f} dB")
        
        # Save the generator model
        try:
            generator.save(os.path.join(trial_dir, 'generator.keras'))
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
        
        # Return average PSNR for optimization
        return final_psnr
        
    except Exception as e:
        # Log the error and continue with next trial
        print(f"Error in trial {trial_number}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0  # Return poor score on error

# Main function to run Optuna optimization
def run_optuna_optimization(n_trials=20, study_name="sr_gan_optimization"):
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create Optuna study for maximizing PSNR
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # We want to maximize PSNR
        sampler=TPESampler(seed=SEED),
        pruner=pruner
    )
    
    # Run optimization
    try:
        study.optimize(objective, n_trials=n_trials)
    except KeyboardInterrupt:
        print("Optimization stopped early by user")
    except Exception as e:
        print(f"Error during optimization: {e}")
        traceback.print_exc()
    
    # Print best parameters
    print("Optimization completed!")
    print("\nBest trial:")
    try:
        best_trial = study.best_trial
        print(f"  PSNR: {best_trial.value:.2f} dB")
        print("  Parameters: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    except Exception as e:
        print(f"Error accessing best trial: {e}")
        print("Falling back to default parameters")
        best_trial = None
    
    # Save study results
    try:
        study_results_file = os.path.join(OUTPUT_DIR, "optuna_results.pkl")
        with open(study_results_file, "wb") as f:
            import pickle
            pickle.dump(study, f)
            
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(OUTPUT_DIR, "optimization_history.png"))
        
        plt.figure(figsize=(12, 8))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(os.path.join(OUTPUT_DIR, "parameter_importance.png"))
    except Exception as e:
        print(f"Error saving study results: {e}")
    
    # Train the final model with the best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    
    # Extract best parameters or use defaults
    if best_trial:
        best_params = best_trial.params
    else:
        print("Using default parameters instead")
        best_params = {
            'learning_rate': LEARNING_RATE,
            'leakyrelu_alpha': LEAKYRELU_ALPHA,
            'num_residual_blocks': NUM_RESIDUAL_BLOCKS,
            'filter_size': 64,
            'dropout_rate': 0.2,
            'weight_decay': 0,
            'batch_size': BATCH_SIZE,
            'd_learning_rate': LEARNING_RATE,
            'l1_weight': 1.0,
            'perceptual_weight': 0.001,
            'label_smoothing': 0,
            'use_relativistic': True
        }
    
    # Create datasets with best batch size
    train_ds, val_ds, test_ds = get_patch_dataset(
        dataset_path, 
        batch_size=best_params.get('batch_size', BATCH_SIZE)
    )
    
    # Convert to distributed datasets
    train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
    val_dist_ds = strategy.experimental_distribute_dataset(val_ds)
    
    # Build final models
    with strategy.scope():
        final_generator, final_discriminator, final_adversarial_model = build_and_compile_models(
            learning_rate=best_params.get('learning_rate', LEARNING_RATE),
            leakyrelu_alpha=best_params.get('leakyrelu_alpha', LEAKYRELU_ALPHA),
            num_residual_blocks=int(best_params.get('num_residual_blocks', NUM_RESIDUAL_BLOCKS)),
            filter_size=best_params.get('filter_size', 64),
            dropout_rate=best_params.get('dropout_rate', 0.2),
            weight_decay=best_params.get('weight_decay', 0)
        )
    
    # Train the final model for more epochs
    final_epochs = EPOCHS  # Use the full number of epochs
    d_losses, g_losses, final_psnr = train_gan(
        final_generator, final_discriminator, final_adversarial_model,
        train_ds, val_ds,
        epochs=final_epochs,
        d_learning_rate=best_params.get('d_learning_rate', best_params.get('learning_rate', LEARNING_RATE)),
        g_learning_rate=best_params.get('learning_rate', LEARNING_RATE),
        l1_weight=best_params.get('l1_weight', 1.0),
        perceptual_weight=best_params.get('perceptual_weight', 0.001),
        label_smoothing=best_params.get('label_smoothing', 0),
        use_relativistic=best_params.get('use_relativistic', True)
    )
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Final Model Training Losses')
    plt.savefig(os.path.join(OUTPUT_DIR, 'final_training_losses.png'))
    
    # Save the optimized generator model
    final_generator.save(os.path.join(OUTPUT_DIR, 'final_sr_generator.keras'))
    print(f"Final model saved with validation PSNR: {final_psnr:.2f} dB")
    
    # Generate and save example results
    print("Generating example results...")
    for lr_batch, hr_batch in test_ds.take(5):
        lr_image = lr_batch[0]
        hr_image = hr_batch[0]

        # Generate SR image
        sr_image = final_generator(tf.expand_dims(lr_image, 0), training=False)[0]

        # Convert images from [-1, 1] to [0, 1] for display
        lr_image = (lr_image + 1) / 2.0
        hr_image = (hr_image + 1) / 2.0
        sr_image = (sr_image + 1) / 2.0

        # Calculate PSNR
        mse = np.mean((hr_image.numpy() - sr_image.numpy()) ** 2)
        if mse > 0:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        else:
            psnr = 100  # Perfect match

        # Plot the images
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('Low Resolution')
        plt.imshow(lr_image)

        plt.subplot(1, 3, 2)
        plt.title(f'Super Resolution (PSNR: {psnr:.2f}dB)')
        plt.imshow(sr_image)

        plt.subplot(1, 3, 3)
        plt.title('High Resolution (Ground Truth)')
        plt.imshow(hr_image)

        plt.savefig(os.path.join(OUTPUT_DIR, f'final_result_{np.random.randint(1000)}.png'))
        plt.close()
    
    return best_params, final_psnr, final_generator

# Enable more verbose TensorFlow logging
def enable_verbose_logging():
    tf.debugging.set_log_device_placement(True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all logs
    print("Enabled verbose TensorFlow logging")

# If executed as script
if __name__ == "__main__":
    import argparse
    
    # Print system information
    print("\n" + "="*80)
    print(f"SYSTEM INFORMATION:")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    print(f"CPU devices: {tf.config.list_physical_devices('CPU')}")
    print("="*80 + "\n")
    
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='SR-GAN training with Optuna')
    parser.add_argument('--trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs for final training')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Default batch size')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory to save results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose TensorFlow logging')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with more print statements')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Update variables based on arguments - no global declaration needed
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    OUTPUT_DIR = args.output_dir
    
    # Enable verbose logging if requested
    if args.verbose:
        enable_verbose_logging()
        
    # Set debug flag for more print statements
    DEBUG = args.debug
    
    print(f"\nStarting SR-GAN training with TensorFlow {tf.__version__}")
    print(f"Dataset path: {dataset_path}")
    print(f"Using batch size: {BATCH_SIZE}, epochs: {EPOCHS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Debug mode: {'ON' if DEBUG else 'OFF'}")
    
    # Print timestamp for tracking
    start_time = time.time()
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dataset structure
    if not check_dataset():
        print("Dataset structure issue detected. Please check the dataset path.")
        import sys
        sys.exit(1)
        
    # Create datasets with patching
    print(f"Creating datasets with patch dimensions: LR={PATCH_LR_HEIGHT}x{PATCH_LR_WIDTH}, HR={PATCH_HR_HEIGHT}x{PATCH_HR_WIDTH}")
    train_ds, val_ds, test_ds = get_patch_dataset(dataset_path, batch_size=BATCH_SIZE)
    
    # Add the validation code:
    print("Validating dataset normalization...")
    for lr_batch, hr_batch in train_ds.take(1):
        print(f"LR range: [{tf.reduce_min(lr_batch).numpy():.2f}, {tf.reduce_max(lr_batch).numpy():.2f}]")
        print(f"HR range: [{tf.reduce_min(hr_batch).numpy():.2f}, {tf.reduce_max(hr_batch).numpy():.2f}]")
        
        # If range is not approximately [-1, 1], there is a preprocessing issue
        if tf.reduce_max(lr_batch).numpy() > 2.0 or tf.reduce_min(lr_batch).numpy() < -2.0:
            print("WARNING: LR data is not properly normalized!")
            print("You should fix the preprocessing function in create_sr_gan_dataset")
        
        if tf.reduce_max(hr_batch).numpy() > 2.0 or tf.reduce_min(hr_batch).numpy() < -2.0:
            print("WARNING: HR data is not properly normalized!")
            print("You should fix the preprocessing function in create_sr_gan_dataset")
    
    # Additional normalization diagnostic
    print("\nDiagnostic test for data normalization:")
    for lr_batch, hr_batch in train_ds.take(1):
        print(f"After processing - LR range: [{tf.reduce_min(lr_batch).numpy():.2f}, {tf.reduce_max(lr_batch).numpy():.2f}]")
        print(f"After processing - HR range: [{tf.reduce_min(hr_batch).numpy():.2f}, {tf.reduce_max(hr_batch).numpy():.2f}]")
        
        # Force normalization on existing dataset if needed
        if tf.reduce_max(lr_batch).numpy() > 1.1 or tf.reduce_min(lr_batch).numpy() < -1.1:
            print("WARNING: Applying emergency normalization to dataset...")
            # This is a backup approach - better to fix the preprocessing functions
            normalized_lr = tf.clip_by_value((lr_batch / 255.0) * 2 - 1, -1.0, 1.0)
            normalized_hr = tf.clip_by_value((hr_batch / 255.0) * 2 - 1, -1.0, 1.0)
            print(f"After emergency normalization - LR range: [{tf.reduce_min(normalized_lr).numpy():.2f}, {tf.reduce_max(normalized_lr).numpy():.2f}]")
            print(f"After emergency normalization - HR range: [{tf.reduce_min(normalized_hr).numpy():.2f}, {tf.reduce_max(normalized_hr).numpy():.2f}]")
    
    # Create optimization study with Optuna
    if args.trials > 0:
        print(f"Starting Optuna hyperparameter optimization with {args.trials} trials...")
        
        # Run Optuna optimization
        try:
            best_params, final_psnr, final_generator = run_optuna_optimization(n_trials=args.trials)
            
            print(f"Optimization complete! Best PSNR: {final_psnr:.2f} dB")
            print("Best parameters:")
            for key, value in best_params.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error during optimization: {e}")
            print("Falling back to default hyperparameters")
            
            # Build models with default parameters under strategy scope
            with strategy.scope():
                generator, discriminator, adversarial_model = build_and_compile_models(
                    learning_rate=LEARNING_RATE,
                    leakyrelu_alpha=LEAKYRELU_ALPHA,
                    num_residual_blocks=NUM_RESIDUAL_BLOCKS
                )
            
            # Train with default parameters
            print(f"Training SR-GAN model for {EPOCHS} epochs with default parameters...")
            
            # Convert to distributed datasets
            train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
            val_dist_ds = strategy.experimental_distribute_dataset(val_ds)
            
            d_losses, g_losses, final_psnr = train_gan(
                generator, discriminator, adversarial_model,
                train_ds, val_ds, 
                epochs=EPOCHS,
                l1_weight=1.0,
                perceptual_weight=0.001  # Reduced perceptual weight
            )
            
            # Plot training losses
            plt.figure(figsize=(10, 5))
            plt.plot(d_losses, label='Discriminator Loss')
            plt.plot(g_losses, label='Generator Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training Losses')
            plt.savefig(os.path.join(OUTPUT_DIR, 'training_losses.png'))
            
            # Save the final model
            generator.save(os.path.join(OUTPUT_DIR, 'sr_generator.keras'))
            print(f"Model saved with validation PSNR: {final_psnr:.2f} dB")
    else:
        # Use default hyperparameters
        print("Using default hyperparameters (no Optuna optimization)")
        
        # Build models with default parameters under strategy scope
        with strategy.scope():
            generator, discriminator, adversarial_model = build_and_compile_models(
                learning_rate=LEARNING_RATE,
                leakyrelu_alpha=LEAKYRELU_ALPHA,
                num_residual_blocks=NUM_RESIDUAL_BLOCKS
            )
        
        # Train the model
        print(f"Training SR-GAN model for {EPOCHS} epochs...")
        
        # Convert to distributed datasets
        train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
        val_dist_ds = strategy.experimental_distribute_dataset(val_ds)
        
        d_losses, g_losses, final_psnr = train_gan(
            generator, discriminator, adversarial_model,
            train_ds, val_ds, 
            epochs=EPOCHS,
            l1_weight=1.0,
            perceptual_weight=0.001  # Reduced perceptual weight
        )
        
        # Plot training losses
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label='Discriminator Loss')
        plt.plot(g_losses, label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Losses')
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_losses.png'))
        
        # Save the final model
        generator.save(os.path.join(OUTPUT_DIR, 'sr_generator.keras'))
        print(f"Model saved with validation PSNR: {final_psnr:.2f} dB")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"SR-GAN training completed successfully!")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
