import numpy as np
import matplotlib.pyplot as plt
import umap


def plot_umap(latent_vectors, labels=None, save_path=None):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
    plt.colorbar()
    plt.title("UMAP Projection of Latent Space")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def visualize_3d(volume, threshold=0.5, save_path=None):
    from mpl_toolkits.mplot3d import Axes3D

    volume_np = volume.squeeze().detach().cpu().numpy()
    filled = volume_np > threshold
    filled_coords = np.argwhere(filled)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(filled_coords[:, 0], filled_coords[:, 1], filled_coords[:, 2], alpha=0.6)

    ax.set_xlabel('Depth')
    ax.set_ylabel('Height')
    ax.set_zlabel('Width')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
