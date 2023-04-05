import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go

# def plot_cluster_subplots(df,n_clusters,main_title='Cluster Plot',):

#     """
#     function to allow us to play around with different feature columns and plot different visualisations in 4 different plots (as seen below)

#     Columns with corresponding feature number
#     'User' = 0, 'Card' = 1, 'Year' = 2, 'Month' = 3, 'Day' = 4,
#     'Time' = 5, 'Amount' = 6, 'Merchant_Name' = 7, 'Zip' = 8,
#     'MCC' = 9, 'Errors_Encoded' = 10, 'Use_Chip_Encoded' = 11,
#     'Merchant_City_Encoded' = 12, 'Merchant_State_Encoded' = 13
#     """

#     colours = ['#3CCF4E', '#1B2430', '#D61C4E', '#3330E4', '#EF5B0C', '#EAE509']
#     fig, axs = plt.subplots(1, 4, figsize=(15, 8))
#     fig.suptitle(main_title)


#     for i in range(0,n_clusters): ## iterating through n clusters

#         filtered_df = df[df['pred_labels'] == i]
#         filtered_df = filtered_df.to_numpy()

#         #Plotting the results
#         plt.subplot(1, 4,1)
#         plt.scatter(filtered_df[:,1] , filtered_df[:,10] , c = colours[i])
#         plt.title('Card Versus Errors')

#         plt.subplot(1, 4,2)
#         plt.scatter(filtered_df[:,5] , filtered_df[:,6] , c = colours[i])
#         plt.title('Time Versus Amount')

#         plt.subplot(1, 4,3)
#         plt.scatter(filtered_df[:,2] , filtered_df[:,13] , c = colours[i])
#         plt.title('Year Versus Merchant State')

#         plt.subplot(1, 4,4)
#         plt.scatter(filtered_df[:,0] , filtered_df[:,10] , c = colours[i])
#         plt.title('User Versus Errors')
#     plt.show()
import plotly.express as px

def plot_cluster_evaluation(inertia, sc_score,k_range,inertia_title, sc_title):
    """Function to evaluate and visualise the inertia score (elbow method) as well as the silhouette coefficient"""

    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    plt.subplot(1, 2,1)
    plt.plot(k_range, inertia,color="#EF5B0C",marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('inertia')
    plt.title(inertia_title)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(k_range, sc_score,color="#FA2FB5",marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.title(sc_title)
    plt.grid(True)

    plt.show()


def plot_cluster_subplots(df,n_clusters,main_title='Cluster Plot',):

    """
    function to allow us to play around with different feature columns and plot different visualisations in 4 different plots (as seen below)

    Columns with corresponding feature number
    'User' = 0, 'Card' = 1, 'Year' = 2, 'Month' = 3, 'Day' = 4,
    'Time' = 5, 'Amount' = 6, 'Merchant_Name' = 7, 'Zip' = 8,
    'MCC' = 9, 'Errors_Encoded' = 10, 'Use_Chip_Encoded' = 11,
    'Merchant_City_Encoded' = 12, 'Merchant_State_Encoded' = 13
    """

    colours = ['#3CCF4E', '#1B2430', '#D61C4E', '#3330E4', '#EF5B0C', '#EAE509']
    fig, axs = plt.subplots(1, 4, figsize=(15, 8))
    fig.suptitle(main_title)


    for i in range(0,n_clusters): ## iterating through n clusters

        filtered_df = df[df['pred_labels'] == i]
        filtered_df = filtered_df.to_numpy()

        #Plotting the results
        plt.subplot(1, 4,1)
        plt.scatter(filtered_df[:,1] , filtered_df[:,10] , c = colours[i])
        plt.title('Card Versus Errors')

        plt.subplot(1, 4,2)
        plt.scatter(filtered_df[:,5] , filtered_df[:,6] , c = colours[i])
        plt.title('Time Versus Amount')

        plt.subplot(1, 4,3)
        plt.scatter(filtered_df[:,2] , filtered_df[:,13] , c = colours[i])
        plt.title('Year Versus Merchant State')

        plt.subplot(1, 4,4)
        plt.scatter(filtered_df[:,0] , filtered_df[:,10] , c = colours[i])
        plt.title('User Versus Errors')

        fig = px.scatter_3d(filtered_df, x=filtered_df[:,1], y=filtered_df[:,5], z=filtered_df[:,2],
                      color=filtered_df[:,10], color_discrete_sequence=colours)

        fig.update_layout(title=f"Card, Time and Year 3D plot for cluster {i}")

        fig.show()




def plot_clusters_3d(X_sc_copy, labels):
    fig = go.Figure()
    for cluster in set(labels):
        fig.add_trace(go.Scatter3d(
            x=X_sc_copy[labels == cluster, 0],
            y=X_sc_copy[labels == cluster, 1],
            z=X_sc_copy[labels == cluster, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=cluster,
                opacity=0.8
            ),
            name=f'Cluster {cluster}'
        ))
    fig.update_layout(
        title='Clusters in 3D space',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )
    fig.show()
