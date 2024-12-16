import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import numpy as np
from PIL import Image
import os
from collections import defaultdict
from layers.Autoformer_EncDec import series_decomp


def merge_images_by_prefix(image_dir, output_dir, debug_frequency, itr=0, max_images_per_row=4):
    """
    Merges images with the same prefix into a grid with a maximum number of images per row and saves them in the output directory.

    Args:
        image_dir (str): Directory containing the images to merge.
        output_dir (str): Directory to save the merged images.
        max_images_per_row (int): Maximum number of images per row.
    """
    if itr % debug_frequency != 0:
        return
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Group images by prefix
    images_by_prefix = defaultdict(list)

    # Collect images by their prefix
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            prefix = filename.split("_")[0]
            images_by_prefix[prefix].append(os.path.join(image_dir, filename))

    # Merge images with the same prefix
    for prefix, image_paths in images_by_prefix.items():
        images = [Image.open(img) for img in image_paths]
        widths, heights = zip(*(img.size for img in images))
        
        # Calculate grid dimensions
        num_images = len(images)
        num_rows = (num_images + max_images_per_row - 1) // max_images_per_row
        max_width = max(widths)
        max_height = max(heights)
        
        # Create a new blank image with the combined size
        merged_image = Image.new("RGB", (max_width * max_images_per_row, max_height * num_rows))
        
        # Paste each image into the grid
        x_offset = 0
        y_offset = 0
        for i, img in enumerate(images):
            merged_image.paste(img, (x_offset, y_offset))
            x_offset += img.width
            if (i + 1) % max_images_per_row == 0:
                x_offset = 0
                y_offset += img.height
        
        # Save the merged image
        output_path = os.path.join(output_dir, f"{prefix}_merged.png")
        merged_image.save(output_path)

    print("Images have been merged and saved in the output directory.")



def plot_heatmap(x_enc, itr , n_cycles, debug_frequency, batch_sample_index = -1, plot_name = 'x_enc'):
    """
    Plots x_enc as a heatmap for given index in batch.

    Args:
        x_enc (torch.Tensor): The input tensor to plot.
    """
    if itr % debug_frequency != 0:
        return
    # Plot x_enc as a heatmap for each batch
    for i in range(x_enc.shape[0]):
        if batch_sample_index != -1 and i != batch_sample_index:
            continue
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(x_enc[i].detach().cpu().numpy(), cmap='viridis', cbar=True)
        plt.title(f'{plot_name} Heatmap - Batch {i}')
        plt.xlabel('Time')
        plt.ylabel('Features')
        # Add red lines for every n_cycle between columns
        for cycle in range(1, x_enc.shape[2] // n_cycles):
            ax.axvline(x=cycle * n_cycles, color='red', linestyle='--')
        plt.savefig(f"{debug_folder}{itr}_{plot_name}_plot.png")
        plt.close()  # Close the figure to avoid memory issues
        # time.sleep(0.1)

def plot_attention(attention_matrix, itr, n_cycles, batch_sample_index=-1):
    """
    Plots the attention weights as a heatmap for a given index in batch.

    Args:
        attention_matrix (torch.Tensor): The attention matrix to plot.
        input_data_with_nan (torch.Tensor): The input data with NaNs to mask the attention weights.
        itr (int): The current iteration number.
        batch_sample_index (int): The index of the batch to plot. If -1, plot all batches.
    """
    if itr % debug_frequency != 0:
        return

    attention_matrix = attention_matrix[0].detach().cpu().numpy()

    # Compute the average attention weights over the heads
    avg_attention_weights = np.mean(attention_matrix, axis=1)

    for batch_index in range(avg_attention_weights.shape[0]):
        if batch_sample_index != -1 and batch_index != batch_sample_index:
            continue

        plt.figure(figsize=(10, 6))
        plt.imshow(avg_attention_weights[batch_index], cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'Attention Matrix (Averaged over Heads) - Batch {batch_index}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')

        # Overlay grid lines
        num_rows, num_cols = avg_attention_weights[batch_index].shape
        # Add red lines for every n_cycle between columns and rows
        # Add red lines for every n_cycle between columns and rows
        for cycle in range(1, num_cols // n_cycles):
            plt.axvline(x=cycle * n_cycles - 0.5, color='red', linestyle='--')
        for cycle in range(1, num_rows // n_cycles):
            plt.axhline(y=cycle * n_cycles - 0.5, color='red', linestyle='--')
        plt.savefig(f"{debug_folder}{itr}_attention_plot.png")
        plt.close()  # Close the figure to avoid memory issues
        # time.sleep(0.1)