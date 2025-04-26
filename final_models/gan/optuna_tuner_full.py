
import os
import optuna
import tensorflow as tf
from gan_model_logged import build_generator, build_discriminator, SimpleTrainer
from vgg_loss_utils import build_vgg_feature_extractor

DATASET_PATH = "/users/smicha17/video-game-sr"

class DatasetManager:
    def __init__(self, data_dir, batch_size=4, lr_patch_size=64, hr_patch_size=256):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = hr_patch_size

    def preprocess(self, lr_path, hr_path):
        lr_image = tf.io.read_file(lr_path)
        lr_image = tf.image.decode_png(lr_image, channels=3)
        lr_image = tf.image.convert_image_dtype(lr_image, tf.float32)
        lr_image = tf.image.resize(lr_image, [self.lr_patch_size, self.lr_patch_size])

        hr_image = tf.io.read_file(hr_path)
        hr_image = tf.image.decode_png(hr_image, channels=3)
        hr_image = tf.image.convert_image_dtype(hr_image, tf.float32)
        hr_image = tf.image.resize(hr_image, [self.hr_patch_size, self.hr_patch_size])

        return lr_image, hr_image

    def create_datasets(self):
        def make_dataset(lr_folder, hr_folder):
            lr_paths = sorted([os.path.join(self.data_dir, lr_folder, f)
                               for f in os.listdir(os.path.join(self.data_dir, lr_folder))])
            hr_paths = sorted([os.path.join(self.data_dir, hr_folder, f)
                               for f in os.listdir(os.path.join(self.data_dir, hr_folder))])
            ds = tf.data.Dataset.from_tensor_slices((lr_paths, hr_paths))
            ds = ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.repeat() 
            ds = ds.batch(self.batch_size)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds

        train_ds = make_dataset("training-lr", "training-hr")
        val_ds = make_dataset("validation-lr", "validation-hr")
        test_ds = make_dataset("test-lr", "test-hr")
        return train_ds, val_ds, test_ds

def objective(trial):
    # Hyperparameter search space
    steps_per_epoch = 1000
    num_blocks = trial.suggest_int("num_residual_blocks", 6, 20)
    filters = trial.suggest_categorical("filters", [32, 64, 96])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3)
    lr_gen = trial.suggest_float("lr_gen", 1e-5, 1e-3, log=True)
    lr_disc = trial.suggest_float("lr_disc", 1e-5, 1e-3, log=True)
    perceptual_wt = trial.suggest_float("perceptual_loss_weight", 1e-3, 1.0, log=True)
    pixel_wt = trial.suggest_float("pixel_loss_weight", 1e-3, 1.0, log=True)
    adv_wt = trial.suggest_float("adv_loss_weight", 1e-3, 1.0, log=True)

    dataset_manager = DatasetManager(DATASET_PATH, batch_size=4)
    train_ds, val_ds, _ = dataset_manager.create_datasets()

    generator = build_generator(num_residual_blocks=num_blocks, filters=filters, dropout_rate=dropout_rate)
    discriminator = build_discriminator(dropout_rate=dropout_rate)

    trainer = SimpleTrainer(
        generator,
        discriminator,
        lr_generator=lr_gen,
        lr_discriminator=lr_disc,
        checkpoint_dir="optuna_trials",
        perceptual_loss_weight=perceptual_wt,
        pixel_loss_weight=pixel_wt,
        adversarial_loss_weight=adv_wt
    )

    history = trainer.train(
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=10,
        steps_per_epoch=steps_per_epoch, 
        start_epoch=1
    )

    return history["val_psnr"][-1] if history["val_psnr"] else 0.0

if __name__ == "__main__":
    print("Starting Optuna Hyperparameter tuning...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best trial:")
    print(study.best_trial)
