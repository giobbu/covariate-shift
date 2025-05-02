def plot_mmd_null_distribution(mmd_perms, mmd, output_path='imgs/mmd_pvalue.png'):
    """
    Plot the null distribution of MMD values from permutations and the observed MMD value.

    Parameters:
    - mmd_perms: List or array of MMD values from permutations.
    - mmd: Observed MMD value.
    - output_path: File path to save the figure.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.hist(mmd_perms, bins=20, alpha=0.7, label='MMD Permutations')
    plt.axvline(mmd, color='red', linestyle='dashed', linewidth=2, label='Observed MMD')
    plt.xlabel('MMD Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of MMD Permutations')
    plt.legend()
    plt.savefig(output_path)
    plt.show()

def plot_bivariate_drift(x_before, x_after, output_path='imgs/covariate_drift.png'):
    """
    Plots scatter plots of feature distributions before and after treatment to visualize covariate drift.

    Parameters:
    - x_before: 2D array-like, features before treatment.
    - x_after: 2D array-like, features after treatment.
    - output_path: String, path to save the figure.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    
    # Before Treatment
    plt.subplot(1, 2, 1)
    plt.scatter(x_before[:, 0], x_before[:, 1], alpha=0.5, label='Before')
    plt.title('Before Treatment')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # After Treatment
    plt.subplot(1, 2, 2)
    plt.scatter(x_after[:, 0], x_after[:, 1], alpha=0.5, label='After', color='orange')
    plt.title('After Treatment')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def plot_pdf_drift(x_before, x_after, output_path='imgs/pdf_drift.png', plot_overlap=False):
    """
    Plots the probability density function (PDF) of features before and after treatment to visualize covariate drift.
    Parameters:
    - x_before: 2D array-like, features before treatment.
    - x_after: 2D array-like, features after treatment.
    - output_path: String, path to save the figure.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    import numpy as np

    # Create a meshgrid for the 3D plot
    x_min = min(x_before[:, 0].min(), x_after[:, 0].min())
    x_max = max(x_before[:, 0].max(), x_after[:, 0].max())
    y_min = min(x_before[:, 1].min(), x_after[:, 1].min())
    y_max = max(x_before[:, 1].max(), x_after[:, 1].max())

    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_range, y_range)

    # Calculate the PDF for both distributions
    kde_before = gaussian_kde(x_before.T)
    Z_before = kde_before(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    kde_after = gaussian_kde(x_after.T)
    Z_after = kde_after(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    # Plotting
    fig = plt.figure(figsize=(12, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z_before, cmap='viridis', alpha=0.7)
    ax1.set_title('PDF Before Treatment')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Density')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, Z_after, cmap='plasma', alpha=0.7)
    ax2.set_title('PDF After Treatment')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_zlabel('Density')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    if plot_overlap:
        
        # Plotting
        fig = plt.figure(figsize=(8, 6))

        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z_before, cmap='viridis', alpha=0.4, label='Before Treatment')
        ax.plot_surface(X, Y, Z_after, cmap='plasma', alpha=0.5, label='After Treatment')
        ax.set_title('PDF Before and After Treatment')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Density')
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='mediumseagreen', label='Before Treatment'),
                        Patch(facecolor='orchid', label='After Treatment')]
        ax.legend(handles=legend_elements)
        plt.tight_layout()
        output_path = output_path.replace('.png', '_overlap.png')
        plt.savefig(output_path)
        plt.show()
