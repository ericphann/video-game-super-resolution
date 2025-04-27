# evaluates the trained model and color corrects the output
import os
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np

from gan_model import build_generator
from train_srgan import DatasetManager

def plot_images(lr_img, sr_img, sr_corrected_img, hr_img, idx, save_dir):
    """Plot and save Low-Res, Super-Resolved, Corrected SR, and High-Res images side by side."""
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(lr_img)
    axs[0].set_title("Low Resolution (Input)")
    axs[0].axis('off')

    axs[1].imshow(sr_img)
    axs[1].set_title("Super Resolution (SRGAN Output)")
    axs[1].axis('off')
    
    axs[2].imshow(sr_corrected_img)
    axs[2].set_title("Corrected SR Output")
    axs[2].axis('off')

    axs[3].imshow(hr_img)
    axs[3].set_title("High Resolution (Ground Truth)")
    axs[3].axis('off')

    plt.suptitle(f"Example {idx}")

    # Save figure
    save_path = os.path.join(save_dir, f"example_{idx}.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")

    plt.close(fig)

def advanced_color_correction(sr_img, lr_img, ground_truth=None):
    """More balanced color correction"""
    # Convert to LAB color space
    sr_lab = cv2.cvtColor(sr_img, cv2.COLOR_RGB2LAB)
    lr_lab = cv2.cvtColor(lr_img, cv2.COLOR_RGB2LAB)
    
    # Extract channels
    sr_l, sr_a, sr_b = cv2.split(sr_lab)
    lr_l, lr_a, lr_b = cv2.split(lr_lab)

    sr_mean = np.mean(sr_l)
    lr_mean = np.mean(lr_l)
    
    # Simple linear scaling based on means
    scaling_factor = lr_mean / sr_mean if sr_mean > 0 else 1.0
    
    sr_l_matched = np.clip(sr_l * scaling_factor * 0.85, 0, 255).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sr_l_enhanced = clahe.apply(sr_l_matched)
    
    sr_corrected_lab = cv2.merge([sr_l_enhanced, sr_a, sr_b])
    sr_corrected = cv2.cvtColor(sr_corrected_lab, cv2.COLOR_LAB2RGB)
    
    sr_corrected = cv2.convertScaleAbs(sr_corrected, alpha=1.1, beta=-5)
    
    return sr_corrected

def evaluate_model(generator_path, data_dir, save_dir, num_samples=5):
    """Load the model and save visualization outputs."""
    print(f"Loading generator from {generator_path}...")
    generator = tf.keras.models.load_model(generator_path, compile=False)

    dataset_manager = DatasetManager(data_dir)
    _, _, test_ds = dataset_manager.create_datasets()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory for saving figures: {save_dir}")

    test_batches = iter(test_ds)

    for i in range(num_samples):
        lr_batch, hr_batch = next(test_batches)
        sr_batch = generator.predict(lr_batch)

        lr_img = lr_batch[0].numpy()
        sr_img = sr_batch[0] # already a numpy array
        hr_img = hr_batch[0].numpy()

        lr_img = (lr_img * 255.0).clip(0, 255).astype('uint8')
        sr_img = ((sr_img + 1.0) / 2.0 * 255.0).clip(0, 255).astype('uint8')  # tanh output
        hr_img = (hr_img * 255.0).clip(0, 255).astype('uint8')
        
        sr_corrected_img = advanced_color_correction(sr_img, lr_img, ground_truth=hr_img)
        # Extra gamma and contrast correction
        gamma = 1.2
        look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        sr_corrected_img = cv2.LUT(sr_corrected_img, look_up_table)

        alpha = 1.1  # contrast
        beta = 2     # brightness
        sr_corrected_img = cv2.convertScaleAbs(sr_corrected_img, alpha=alpha, beta=beta)

        # sharpening
        sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        # apply sharpening
        sr_corrected_img = cv2.filter2D(sr_corrected_img, -1, sharpen_kernel)

        plot_images(lr_img, sr_img, sr_corrected_img, hr_img, idx=i+1, save_dir=save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_path", type=str, required=True, help="Path to the saved generator model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--save_dir", type=str, default="./eval_outputs", help="Directory to save evaluation images")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of examples to generate")
    args = parser.parse_args()

    evaluate_model(args.generator_path, args.data_dir, args.save_dir, args.num_samples)
