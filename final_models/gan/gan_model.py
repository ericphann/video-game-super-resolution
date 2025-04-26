import os
import tensorflow as tf
from tensorflow.keras import layers, models
from vgg_loss_utils import build_vgg_feature_extractor, compute_perceptual_loss

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

LR_PATCH_SIZE = 64
HR_PATCH_SIZE = 256

def build_generator(leakyrelu_alpha=0.2, num_residual_blocks=8, filters=64,
                    lr_patch_size=LR_PATCH_SIZE, dropout_rate=0.0):
    print(f"Building generator with {num_residual_blocks} residual blocks, {filters} filters, dropout={dropout_rate}...")
    inputs = layers.Input(shape=(lr_patch_size, lr_patch_size, 3))
    x = layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.LeakyReLU(negative_slope=leakyrelu_alpha)(x)
    skip_connection = x

    def residual_block(x):
        res = layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        res = layers.LeakyReLU(negative_slope=leakyrelu_alpha)(res)
        if dropout_rate > 0.0:
            res = layers.Dropout(dropout_rate)(res)
        res = layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(res)
        return layers.Add()([x, res])

    for _ in range(num_residual_blocks):
        x = residual_block(x)

    x = layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.Add()([x, skip_connection])

    for _ in range(2):
        x = layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU(negative_slope=leakyrelu_alpha)(x)

    outputs = layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh', kernel_initializer='glorot_uniform')(x)
    return models.Model(inputs, outputs)

def build_discriminator(leakyrelu_alpha=0.2, hr_patch_size=HR_PATCH_SIZE, dropout_rate=0.0):
    print(f"Building discriminator with dropout={dropout_rate}...")
    inputs = layers.Input(shape=(hr_patch_size, hr_patch_size, 3))
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.LeakyReLU(negative_slope=leakyrelu_alpha)(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)

    for filters, stride in [(64, 2), (128, 1), (128, 2), (256, 1)]:
        x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=leakyrelu_alpha)(x)
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU(negative_slope=leakyrelu_alpha)(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
    return models.Model(inputs, outputs)

class SimpleTrainer:
    def __init__(self, generator, discriminator,
                 lr_generator=1e-4, lr_discriminator=1e-5,
                 checkpoint_dir='checkpoints',
                 perceptual_loss_weight=1.0,
                 pixel_loss_weight=1.0,
                 adversarial_loss_weight=1.0):

        self.generator = generator
        self.discriminator = discriminator
        self.lr_generator = lr_generator
        self.lr_discriminator = lr_discriminator
        self.checkpoint_dir = checkpoint_dir

        ensure_dir(self.checkpoint_dir)

        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_generator, beta_1=0.9)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_discriminator, beta_1=0.9)

        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.vgg_model = build_vgg_feature_extractor(input_shape=(None, None, 3))

        self.perceptual_loss_weight = perceptual_loss_weight
        self.pixel_loss_weight = pixel_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight

        print("Trainer initialized successfully")

    @tf.function
    def train_step(self, lr_batch, hr_batch):
        tf.print("Entering train_step")

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_hr = self.generator(lr_batch, training=True)
            tf.print("Generator output computed")

            real_output = self.discriminator(hr_batch, training=True)
            fake_output = self.discriminator(fake_hr, training=True)
            tf.print("Discriminator outputs computed")

            adversarial_loss = self.bce_loss(tf.ones_like(fake_output), fake_output)
            tf.print("Adversarial loss computed")

            perceptual_loss = compute_perceptual_loss(self.vgg_model, hr_batch, fake_hr)
            tf.print("Perceptual loss computed")

            pixel_loss = tf.reduce_mean(tf.abs(hr_batch - fake_hr))
            tf.print("Pixel loss computed")

            gen_total_loss = (
                self.perceptual_loss_weight * perceptual_loss +
                self.pixel_loss_weight * pixel_loss +
                self.adversarial_loss_weight * adversarial_loss
            )

            disc_loss_real = self.bce_loss(tf.ones_like(real_output), real_output)
            disc_loss_fake = self.bce_loss(tf.zeros_like(fake_output), fake_output)
            disc_loss = disc_loss_real + disc_loss_fake
            tf.print("All losses computed")

        gen_grads = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
        tf.print(" Gradients applied")

        return {
            "gen_loss": gen_total_loss,
            "disc_loss": disc_loss,
            "pixel_loss": pixel_loss,
            "perceptual_loss": perceptual_loss,
            "adv_loss": adversarial_loss
        }

    def train(self, train_ds, val_ds=None, epochs=10, steps_per_epoch=1000, start_epoch=1):
        history = {
            "gen_loss": [],
            "disc_loss": [],
            "pixel_loss": [],
            "perceptual_loss": [],
            "adv_loss": [],
            "val_psnr": [],
            "val_pixel_loss": []
        }

        for epoch in range(start_epoch, start_epoch + epochs):
            print(f"\nEpoch {epoch}/{start_epoch + epochs - 1}")

            step_count = 0
            for step, (lr_batch, hr_batch) in enumerate(train_ds.take(steps_per_epoch)):
                try:
                    print(f"[Step {step}] LR shape: {lr_batch.shape}, HR shape: {hr_batch.shape}")
                    losses = self.train_step(lr_batch, hr_batch)
                    step_count += 1
                except Exception as e:
                    print(f"[ERROR @ Step {step}] {str(e)}")
                    continue

            for k in losses:
                history[k].append(float(losses[k]))

            if val_ds is not None:
                psnrs, val_pix_losses = [], []
                for val_lr, val_hr in val_ds.take(100):
                    try:
                        sr = self.generator(val_lr, training=False)
                        if sr.shape == val_hr.shape:
                            psnr = tf.image.psnr(sr, val_hr, max_val=1.0)
                            pix_loss = tf.reduce_mean(tf.abs(val_hr - sr))
                            psnrs.extend(psnr.numpy().tolist())
                            val_pix_losses.append(pix_loss.numpy())
                        else:
                            print(f"[WARN] Skipped mismatched batch. SR shape: {sr.shape}, HR shape: {val_hr.shape}")
                    except Exception as e:
                        print(f"[VAL ERROR] Skipping batch: {e}")
                        continue

                avg_psnr = tf.reduce_mean(psnrs).numpy() if psnrs else 0.0
                avg_pix = tf.reduce_mean(val_pix_losses).numpy() if val_pix_losses else 0.0
                history["val_psnr"].append(avg_psnr)
                history["val_pixel_loss"].append(avg_pix)
                print(f"[Epoch {epoch}] val_psnr: {avg_psnr:.4f}, val_pixel_loss: {avg_pix:.4f}")

            print(f"[Epoch {epoch}] Completed {step_count} training steps.")

            # Save the generator after every epoch
            model_save_path = os.path.join(self.checkpoint_dir, f"generator_epoch_{epoch}.h5")
            self.generator.save(model_save_path)
            print(f"[Epoch {epoch}] Generator model saved to {model_save_path}")

        # Save the final generator model
        final_model_save_path = os.path.join(self.checkpoint_dir, "generator_final.h5")
        self.generator.save(final_model_save_path)
        print(f"Final generator model saved to {final_model_save_path}")

        return history
