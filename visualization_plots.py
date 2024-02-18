import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import ceil, floor, sqrt

# Plotting colors
colors = [
    "coral",
    "darkcyan",
    "firebrick",
    "goldenrod",
    "darkolivegreen",
    "saddlebrown",
    "plum",
    "khaki",
    "purple",
]

glass_type = {
    1: "building_windows_float_processed",
    2: "building_windows_non_float_processed",
    3: "vehicle_windows_float_processed",
    4: "vehicle_windows_non_float_processed (none in this database)",
    5: "containers",
    6: "tableware",
    7: "headlamps",
}


def pca_scatter_2d(X):
    """
    Visualizes the 2D projection of a dataset using PCA components.

    Parameters:
    - X: pandas DataFrame containing PCA1, PCA2, and Type columns.

    The function creates and displays a scatter plot of the first two principal components,
    color-coded by the Type column to indicate different categories within the data.
    """

    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        x="PCA1",
        y="PCA2",
        hue="Type",
        palette="tab10",
        data=X,
    )

    plt.title("PCA of Glass Dataset")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Type of Glass")
    plt.show()


def plot_pca_3d(pca_trans):
    """
    Visualizes the 3D projection of a dataset using PCA components.

    Parameters:
    - pca_trans: numpy.ndarray
        The transformed dataset after applying PCA.

    The function creates and displays a 3D scatter plot of the first three principal components.
    """
    fig = plt.figure()
    ax_3d = fig.add_subplot(projection="3d")
    ax_3d.scatter(
        pca_trans[:, 0],
        pca_trans[:, 1],
        pca_trans[:, 2],
        marker=".",
        s=2,  # Size of datapoints
    )

    plt.show()


def scatter(data, plot_n, marker_size=2):
    """
    Visualizes the scatter plot of the first n features in the dataset.

    Parameters:
    - data: pandas DataFrame
        The dataset to be visualized.

    - plot_n: int
        The number of features to be visualized in the scatter plot.

    - marker_size: int
        The size of the markers in the scatter plot.

    The function creates and displays a scatter plot for each of the first n features in the dataset.
    """

    n = len(data.columns[0:plot_n])
    ncols = 4
    nrows = 4
    nplots_per_fig = ncols * nrows
    nfigs = ceil(n / nplots_per_fig)  # Total number of figures needed

    for i, feature in enumerate(data.columns[0:plot_n]):
        fig_num = i // nplots_per_fig  # Determine the current figure number
        plot_num = (
            i % nplots_per_fig
        )  # Determine the subplot index within the current figure

        if plot_num == 0:  # Create a new figure if starting a new set of subplots
            fig, axs = plt.subplots(nrows, ncols, figsize=(20, 8))
            axs = axs.flatten()

        data_arr = np.asarray(data[feature])
        axs[plot_num].plot(
            data_arr, "o", markersize=marker_size, color=colors[i % len(colors)]
        )
        axs[plot_num].set_title(feature)
        axs[plot_num].set_xlabel("Feature value")

        plt.tight_layout()

        # Hide any unused subplots in the last figure
        if i == n - 1:  # Check if this is the last plot
            for j in range(plot_num + 1, nplots_per_fig):
                axs[j].set_visible(False)

    plt.show()  # Display all figures at once after creating all of them


def plot_pca_exp_var(pca_var, threshold=0.9):
    """
    Visualizes the variance explained by the principal components.

    Parameters:
    - pca_var: numpy.ndarray
        The variance explained by each principal component.

    - threshold: float

    The function creates and displays a line plot of the variance explained by each principal component,
    as well as the cumulative variance explained by all principal components.

    The threshold parameter is used to display a horizontal line indicating the threshold for cumulative variance.
    """
    fig = plt.figure()
    ax_var = fig.add_subplot()
    ax_var.plot(pca_var, "o--")
    ax_var.plot(np.cumsum(pca_var), "x--")
    ax_var.plot([threshold, threshold], "k--")
    plt.xlabel("Principal component")
    plt.ylabel("Variance explained")
    plt.legend(["Individual", "Cumulative", "Threshold"])
    plt.grid()
    plt.show()


def plot_pca_3d(pca_trans):
    """
    Visualizes the 3D projection of a dataset using PCA components.

    Parameters:
    - pca_trans: numpy.ndarray
        The transformed dataset after applying PCA.

    The function creates and displays a 3D scatter plot of the first three principal components.
    """
    fig = plt.figure()
    ax_3d = fig.add_subplot(projection="3d")
    ax_3d.scatter(
        pca_trans[:, 0],
        pca_trans[:, 1],
        pca_trans[:, 2],
        marker=".",
        s=2,  # Size of datapoints
    )

    plt.show()


def histogram(df):
    """
    Visualizes the distribution of each attribute in the dataset using histograms.

    Parameters:
    - df: pandas DataFrame
        The dataset to be visualized.

    The function creates and displays a histogram for each attribute in the dataset.
    """
    # Set the aesthetics for the plots
    sns.set_theme(style="whitegrid")

    # Plotting distributions for each attribute to check for outliers and distribution shape
    fig, axs = plt.subplots(3, 3, figsize=(15, 9))

    for i, col in enumerate(df.columns):
        # Calculate row and column index for each subplot
        row = i // 3
        col_idx = i % 3

        # Histogram plotting
        sns.histplot(df[col], kde=True, ax=axs[row, col_idx], colors=colors[i])
        axs[row, col_idx].set_title(f"Distribution of {col}")

    plt.tight_layout()
    plt.show()


def boxplot(X):
    """
    Visualizes the distribution of each attribute in the dataset using boxplots.

    Parameters:
    - X: pandas DataFrame
        The dataset to be visualized.

    The function creates and displays a boxplot for each attribute in the dataset.
    """
    sns.set_theme(style="whitegrid")

    fig, axs = plt.subplots(3, 3, figsize=(15, 9))

    for i, col in enumerate(df.columns):

        row = i // 3
        col_inx = i % 3
        print(i)

        sns.boxplot(
            x=df[col], ax=axs[row, col_inx], color=colors[i]
        )  # , color=colors[i]
        axs[row, col_inx].set_title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.show()


def correlation_heatmap(df):
    """
    Visualizes the correlation matrix of the dataset.

    Parameters:
    - df: pandas DataFrame
        The dataset to be visualized.

    The function creates and displays a heatmap of the correlation matrix for the dataset.
    """
    corr = df.corr()

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation matrix")
    plt.xlabel("Attributes")
    plt.ylabel("Attributes")

    plt.show()


def loadings_plot(pca, pca_tranform, features, df_type):
    """
    Visualizes the PCA loading plot.

    Parameters:
    - pca: PCA object
        The fitted PCA object.

    - attribute_names: list
        The list of original feature names.

    The function creates and displays a loading plot for the PCA components.
    """
    # Scatter plot of the first two principal components

    # Assuming `pca` is your fitted PCA object and `features` is the list of original feature names
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    plt.figure(figsize=(8, 8))

    sns.scatterplot(
        x="PCA1",
        y="PCA2",
        hue=df_type,
        palette="tab10",
        data=pca_tranform,
        markers=["X"],
    )

    for i in range(loadings.shape[0]):  # Iterate over the number of features
        # Draw the loading vectors
        plt.quiver(
            0,
            0,
            loadings[i, 0],
            loadings[i, 1],
            angles="xy",
            scale_units="xy",
            scale=0.25,
            # color=colors[i % len(colors)],
            color="black",
            width=0.004,
        )

        # Manual adjustments for the overlapping labels of features at indices 4 and 5
        if i == 4:  # Adjusting the first overlapping feature
            plt.text(
                loadings[i, 0] * 5,
                loadings[i, 1] * 4.5,
                features[i],  # + f" ({colors[i]})",
                ha="right",
                va="center",
                color="black",
            )
        elif i == 5:  # Adjusting the second overlapping feature
            plt.text(
                loadings[i, 0] * 4.1,
                loadings[i, 1] * 7,
                features[i],  # + f" ({colors[i]})",
                ha="center",
                va="bottom",
                color="black",
            )
        else:
            plt.text(
                loadings[i, 0] * 4.5,
                loadings[i, 1] * 4.5,
                features[i],
                ha="center",
                va="center",
                color="black",
            )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Loading Plot")
    plt.axis("equal")
    plt.grid()

    # Adjust limits if necessary
    plt.xlim(-5, 5)
    plt.ylim(-2, 4)
    plt.show()


if __name__ == "__main__":
    print(
        "This is a module with visualization functions. Import it to use the functions."
    )
