import matplotlib.pyplot as plt
import numpy as np
from dataloader import BraTSDataset


def visualize_volume_slices(volume, num_slices=9):
    fig, axes = plt.subplots(1, num_slices, figsize=(18, 6))
    slice_indices = np.linspace(0, volume.shape[1] - 1, num_slices).astype(int)

    for i, slice_idx in enumerate(slice_indices):
        axes[i].imshow(volume[0, slice_idx, :, :], cmap="gray")
        axes[i].axis('off')
        axes[i].set_title(f"Slice {slice_idx}")

    plt.show()


def main():
    dataset = BraTSDataset(root_dir="../data/datasets/brats2023/")

    # Random sample from dataset
    sample_idx = np.random.randint(len(dataset))
    volume = dataset[sample_idx]

    # Data details
    print("Sample Data Details:")
    print(f"Shape: {volume.shape}")  # [C, D, H, W]
    print(f"Min value: {volume.min().item()}")
    print(f"Max value: {volume.max().item()}")

    # Visualize slices
    visualize_volume_slices(volume)


if __name__ == "__main__":
    main()
