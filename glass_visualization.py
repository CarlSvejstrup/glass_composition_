import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from math import ceil, floor, sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from visualization_plots import *


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

# # Plotting colors
# colors = [
#     "lime",
#     "red",
#     "yellow",
#     "darkorange",
#     "gold",
#     "cyan",
#     "blue",
#     "violet",
#     "slategrey",
# ]

### Read data ###
df = pd.read_csv(filename, names=attribute_names)
df_type = df.Type
# Storing type in different DF

# Revoming Type and Id
df.drop(["Id", "Type"], axis=1, inplace=True)

# Check for dubplicates
# print(df.duplicated().sum())
# Remove duplicates
# df.drop_duplicates(inplace=True)

# Check for Nan and null values
# print(df.isna().sum(), df.isnull().sum())

### Summary statistics ###
summary_statistics = df.describe()
# print(summary_statistics)


### Standadizing ###
standardize = StandardScaler()
X_standard = standardize.fit_transform(df)
X_standard = pd.DataFrame(X_standard, columns=df.columns)

#### PCA ####
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standard)
df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
explained_variance_ratio = pca.explained_variance_ratio_

df_pca_full = pd.concat([df_pca, df_type], axis=1)


####  Plots form 'visualization_plots.py' #####

# pca_scatter_sns(df_pca_full)
# scatter(df, len(df.columns)
# plot_pca_exp_var(explained_variance_ratio, threshold=0.9):
# plot_pca_3d(X_pca)
# boxplot(df)
# histogram(df)
# correlation_heatmap(df)
# loadings_plot(pca, df_pca, attribute_names[1:-1], df_type)


