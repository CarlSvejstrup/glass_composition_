import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from math import ceil, floor, sqrt
import matplotlib.pyplot as plt
import seaborn as sns

# All atributes exept Type (nomial) are continuos ratio.
attribute_names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K ", "Ca", "Ba", "Fe", "Type"]
filename = "./glass.data"

glass_type = {
    1: "building_windows_float_processed",
    2: "building_windows_non_float_processed",
    3: "vehicle_windows_float_processed",
    4: "vehicle_windows_non_float_processed (none in this database)",
    5: "containers",
    6: "tableware",
    7: "headlamps",
}

# Read data
df = pd.read_csv(filename, names=attribute_names)
df_type = df.Type
# Storing type in different DF


# Revoming Type and Id
df.drop(["Id", "Type"], axis=1, inplace=True)

# Plotting colors
colors = [
    "lime",
    "red",
    "yellow",
    "darkorange",
    "gold",
    "cyan",
    "blue",
    "violet",
    "slategrey",
]


def scatter(data, plot_n, marker_size=2):
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
        axs[plot_num].set_xlabel("Email")

        plt.tight_layout()

        # Hide any unused subplots in the last figure
        if i == n - 1:  # Check if this is the last plot
            for j in range(plot_num + 1, nplots_per_fig):
                axs[j].set_visible(False)

    plt.show()  # Display all figures at once after creating all of them


# scatter(df, len(df.columns))


def plot_pca_exp_var(pca_var, threshold=0.9):
    fig = plt.figure()
    ax_var = fig.add_subplot()
    ax_var.plot(pca_var, "o--")
    ax_var.plot(np.cumsum(pca_var), "x--")
    # ax_var.plot([threshold, threshold], "k--")
    plt.xlabel("Principal component")
    plt.ylabel("Variance explained")
    plt.legend(["Individual", "Cumulative", "Threshold"])
    plt.grid()
    plt.show()


# plot_pca_exp_var(var_ratio_std, 95)


def plot_pca_3d(pca_trans):
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


histogram(df)


def boxplot(X):
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


# print(df.shape)

# boxplot(df)

# Calculate basic summary statistics for the attributes excluding the 'Type of Glass' columns
summary_statistics = df.describe()
# print(summary_statistics)


# Standadizing the data
standardize = StandardScaler()
X_standard = standardize.fit_transform(df)
X_standard = pd.DataFrame(X_standard, columns=df.columns)

# PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_standard)
df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2", "PCA3"])
explained_variance_ratio = pca.explained_variance_ratio_

df_pca_full = pd.concat([df_pca, df_type], axis=1)
# print(df_pca_full)


def pca_scatter_sns(X):
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


# pca_scatter_sns(df_pca_full)


def plot_pca_3d(pca_trans):
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


# plot_pca_exp_var(X_var)
# plot_pca_3d(X_pca)
