import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
matplotlib.use('TkAgg')  # Use the TkAgg backend for interactive plotting
from domain_embeddings import prepare_dataset, get_embedding

def visualize_vectors_with_pca(data, labels=None, n_components=2):
    """
    Perform PCA on the given vector data and visualize it in 2D or 3D.

    Parameters:
        data (numpy.ndarray): The input vector data of shape (n_samples, n_features).
        labels (list or numpy.ndarray, optional): Labels for the data points. Default is None.
        n_components (int): Number of PCA components (2 or 3 for visualization).

    Returns:
        None: Displays the PCA plot.
    """
    if n_components not in [2, 3]:
        raise ValueError("n_components must be 2 or 3 for visualization.")

    # Perform PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)

    # Create the plot
    fig = plt.figure()
    scatter = None  # Ensure scatter is defined for both cases

    if n_components == 2:
        scatter = plt.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=labels,
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='k'
        )
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Visualization (2D)')
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            reduced_data[:, 2],
            c=labels,
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='k'
        )
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title('PCA Visualization (3D)')

    # Add a colorbar if labels are provided
    if labels is not None and scatter is not None:
        cbar = plt.colorbar(scatter, label='Labels')
        cbar.ax.set_title('Label Categories')

    plt.show()


if __name__ == "__main__":
    #data = np.random.rand(100, 5)    
    #labels = np.random.randint(0, 3, size=100)
    chunks, questions = prepare_dataset()
    assert len(chunks) == len(questions)
    data, labels, color_id = [], [], 0
    for i in range(len(chunks)):        
        data.append( get_embedding(chunks[i]) )
        labels.append(color_id)        
        for question in questions[i]:
            data.append( get_embedding(question) )
            labels.append(color_id)                    
        color_id+=1

    visualize_vectors_with_pca(data, labels=labels, n_components=2)
    #visualize_vectors_with_pca(data, labels=labels, n_components=3)
