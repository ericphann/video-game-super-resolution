import os
import tensorflow as tf
import argparse

from gan_model import build_generator, build_discriminator, SimpleTrainer

def load_best_params(path):
    params = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line and 'Parameters' not in line and not line.startswith('Best Trial') and not line.startswith('PSNR'):
                key, val = line.split(':')
                key = key.strip()
                val = val.strip()
                try:
                    val = int(val)
                except ValueError:
                    val = float(val)
                params[key] = val
    return params

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class DatasetManager:
    def __init__(self, data_dir, batch_size=4):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def preprocess(self, lr_path, hr_path):
        lr_image = tf.io.read_file(lr_path)
        lr_image = tf.image.decode_png(lr_image, channels=3)
        lr_image = tf.image.convert_image_dtype(lr_image, tf.float32)
        lr_image = tf.image.resize(lr_image, [64, 64])

        hr_image = tf.io.read_file(hr_path)
        hr_image = tf.image.decode_png(hr_image, channels=3)
        hr_image = tf.image.convert_image_dtype(hr_image, tf.float32)
        hr_image = tf.image.resize(hr_image, [256, 256]) 

        return lr_image, hr_image

    def create_datasets(self):
        def make_dataset(lr_folder, hr_folder):
            lr_paths = sorted([os.path.join(self.data_dir, lr_folder, f)
                               for f in os.listdir(os.path.join(self.data_dir, lr_folder))])
            hr_paths = sorted([os.path.join(self.data_dir, hr_folder, f)
                               for f in os.listdir(os.path.join(self.data_dir, hr_folder))])
            ds = tf.data.Dataset.from_tensor_slices((lr_paths, hr_paths))
            ds = ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            return ds

        train_ds = make_dataset("training-lr", "training-hr")
        val_ds = make_dataset("validation-lr", "validation-hr")
        test_ds = make_dataset("test-lr", "test-hr")
        return train_ds, val_ds, test_ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_best_params", action="store_true", help="Use best Optuna parameters")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="/users/smicha17/checkpoints/manual_run")
    args = parser.parse_args()
    
    data_dir = "/users/smicha17/video-game-sr"
    batch_size = 4
    epochs = args.epochs
    steps_per_epoch = args.steps_per_epoch
    checkpoint_dir = args.checkpoint_dir

    if args.use_best_params:
        best_params = load_best_params("optuna_trials/best_trial_final.txt")
        num_blocks = best_params["num_residual_blocks"]
        filters = int(best_params["filters"])
        dropout = best_params["dropout_rate"]
        lr_gen = best_params["lr_gen"]
        lr_disc = best_params["lr_disc"]
        perceptual_loss_weight = best_params["perceptual_loss_weight"]
        pixel_loss_weight = best_params["pixel_loss_weight"]
        adv_loss_weight = best_params["adv_loss_weight"]
    else:
        # Default values if not using best params
        num_blocks = 16
        filters = 64
        dropout = 0.2
        lr_gen = 1e-4
        lr_disc = 1e-5
        perceptual_loss_weight = 1.0
        pixel_loss_weight = 1.0
        adv_loss_weight = 1.0

    ensure_dir(checkpoint_dir)

    dataset_manager = DatasetManager(data_dir, batch_size=batch_size)
    train_ds, val_ds, test_ds = dataset_manager.create_datasets()

    generator = build_generator(
        num_residual_blocks=num_blocks,
        filters=filters,
        dropout_rate=dropout
    )

    discriminator = build_discriminator(dropout_rate=dropout)

    trainer = SimpleTrainer(
        generator,
        discriminator,
        lr_generator=lr_gen,
        lr_discriminator=lr_disc,
        checkpoint_dir=checkpoint_dir,
        perceptual_loss_weight=perceptual_loss_weight,
        pixel_loss_weight=pixel_loss_weight,
        adversarial_loss_weight=adv_loss_weight
    )

    history = trainer.train(
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        start_epoch=1
    )
