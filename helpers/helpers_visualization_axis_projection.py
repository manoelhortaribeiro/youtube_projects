import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
Compute the projection score for each channel

PARAMETER:
    - axis_vector: Vector representing the axis.
    - channel_vector: Vector representing the channel
RETURN:
    - The projection score
'''
def compute_projection(axis_vector, channel_vector):
    return np.dot(axis_vector, channel_vector)

'''
Enable to plot and save the distribution of the scores stored in df_gender_projection

PARAMETER:
    - df_gender_projection: DataFrame (name of the channel, score)
    - seed_name: pair corresponding to the name of the seed
    - color: String correponding to the Colormaps color

'''
def visualization(df_gender_projection, seed_name, color, nb_for_color, title, save_path = None):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting a histogram of the scores with 30 equal-witdh bins
    n, bins, patches = ax.hist(df_gender_projection['projection'], bins=50, color='green')
    ax.set_ylabel('#Channels')
    ax.set_xlabel('Projection score')

    # This is  the colormap I'd like to use.
    cm = plt.cm.get_cmap(color)

    for i, p in enumerate(patches):
        plt.setp(p, 'facecolor', cm(i/nb_for_color)) 


    # Annotate the seed pair
    x_position = int(list(df_gender_projection[df_gender_projection['name'] == seed_name[0]]['projection'])[0])
    y_position = ax.get_ylim()[1]/10
    ax.annotate(seed_name[0], (x_position, 0), (x_position, y_position), arrowprops = dict(arrowstyle="->"), horizontalalignment='center')


    x_position = int(list(df_gender_projection[df_gender_projection['name'] == seed_name[1]]['projection'])[0])
    y_position = ax.get_ylim()[1]/10
    ax.annotate(seed_name[1], (x_position, 0), (x_position, y_position), arrowprops = dict(arrowstyle="->"), horizontalalignment='center')

    ax.legend()
    ax.set_title(title)
    if save_path != None:
        plt.savefig(save_path)
    plt.show()
    
    
def create_projection(EMBEDDING, axis_vector, dict_idx_name):
    df_projection = pd.DataFrame({'name': EMBEDDING.apply(lambda row: dict_idx_name[row.name], axis = 1)})
    df_projection['projection'] = EMBEDDING.apply(lambda channel_vector: compute_projection(axis_vector, channel_vector), axis = 1)
    return df_projection



def create_plot(df_left, df_right, selected_pairs, title, size, cm, resize = 1, save_path = None):
    fig, ax = plt.subplots(figsize = size)
    ax.axis('off')
    X = np.arange(selected_pairs)
    for ind in X:
        ax.annotate(df_left['name'].iloc[ind], (0, 0), (df_left['projection'].iloc[ind] - 1, ind-0.05), horizontalalignment='right')
        ax.annotate(df_right['name'].iloc[ind], (0, 0), (df_right['projection'].iloc[ind] + 1, ind-0.05), horizontalalignment='left')

        ax.annotate(round(df_left['projection'].iloc[ind], 2), (0, 0), (-0.1, ind-0.05), horizontalalignment='right', color = 'w')
        ax.annotate(round(df_right['projection'].iloc[ind], 2), (0, 0), (0.1, ind-0.05), horizontalalignment='left', color = 'w')

    ax.barh(X, np.array(df_left['projection'])*resize, color = cm(10))
    ax.barh(X, np.array(df_right['projection'])*resize, color = cm(300))
    ax.set_title(title)
    if save_path != None:
        plt.savefig(save_path)
    plt.show()
    