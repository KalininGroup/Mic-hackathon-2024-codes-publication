import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import h5py
import matplotlib.patches as patches
import os
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering



def load_data(filedir):
    composition_tags = [0,7,10,13,20]    #Sm composition %
    img_filename = ['Sm_0_1_HAADF.h5',
                    'Sm_7_2_HAADF.h5',
                    'Sm_10_1_HAADF.h5',
                    'Sm_13_0_HAADF.h5',
                    'Sm_20_0_HAADF.h5',]

    imnum = len(img_filename)
    UCparam_filename = ['Sm_0_1_UCParameterization.h5',
                        'Sm_7_2_UCParameterization.h5',
                        'Sm_10_1_UCParameterization.h5',
                        'Sm_13_0_UCParameterization.h5',
                        'Sm_20_0_UCParameterization.h5',]

    UCparam = []
    for x in UCparam_filename:
        temp = h5py.File(os.path.join(filedir, x), 'r')
        UCparam.append(temp)
        
    imgdata = []
    for x in img_filename:
        temp = h5py.File(os.path.join(filedir, x), 'r')['MainImage']
        imgdata.append(temp)

    SBFOdata = []     #this will be the output list of dictionaries for each dataset

    for i in np.arange(imnum):
        temp_dict = {'Index': i}
        temp_dict['Composition'] = composition_tags[i]
        temp_dict['Image'] = imgdata[i]
        temp_dict['Filename'] = img_filename[i]

        for k in UCparam[i].keys():       #add labels for UC parameterization
            temp_dict[k] = UCparam[i][k][()]

        #select values mapped to ab grid
        temp_dict['ab_a'] = map2grid(UCparam[i]['ab'][()].T, UCparam[i]['ab'][()].T[:,0])[0]       #a array
        temp_dict['ab_b'] = map2grid(UCparam[i]['ab'][()].T, UCparam[i]['ab'][()].T[:,1])[0]       #b array
        temp_dict['ab_x'] = map2grid(UCparam[i]['ab'][()].T, UCparam[i]['xy_COM'][()].T[:,0])[0]   #x array
        temp_dict['ab_y'] = map2grid(UCparam[i]['ab'][()].T, UCparam[i]['xy_COM'][()].T[:,1])[0]   #y array
        temp_dict['ab_Px'] = map2grid(UCparam[i]['ab'][()].T, UCparam[i]['Pxy'][0])[0]             #Px array
        temp_dict['ab_Py'] = map2grid(UCparam[i]['ab'][()].T, UCparam[i]['Pxy'][1])[0]        #Py array
        temp_dict['Vol'] = map2grid(UCparam[i]['ab'][()].T, UCparam[i]['Vol'])[0]     #Vol array

        SBFOdata.append(temp_dict)

    return UCparam, imgdata, SBFOdata


def map2grid(inab, inVal):

  default_val = np.nan
  abrng = [int(np.min(inab[:,0])), int(np.max(inab[:,0])), int(np.min(inab[:,1])), int(np.max(inab[:,1]))]
  abind = inab
  abind[:,0] -= abrng[0]
  abind[:,1] -= abrng[2]
  Valgrid = np.empty((abrng[1]-abrng[0]+1,abrng[3]-abrng[2]+1))
  Valgrid[:] = default_val
  Valgrid[abind[:,0].astype(int),abind[:,1].astype(int)]=inVal[:]
  return Valgrid, abrng



def plot_polarization_vectors(k):
    X = k['ab_x'].ravel()
    Y = k['ab_y'].ravel()
    U = k['ab_Px'].ravel()
    V = k['ab_Py'].ravel()
    Pmag = np.sqrt(U**2 + V**2)
    Pdir = np.arctan2(V, U)
    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = cm.hsv  
    cP = cmap(norm(Pdir))
    return Y, X, U, V, cP


def custom_subimages(imgdata, step_size, window_size):
    # Generate coordinates based on step size
    height, width = imgdata.shape[:2]
    x_coords = np.arange(0, height - window_size[0] + 1, step_size)
    y_coords = np.arange(0, width - window_size[1] + 1, step_size)
    coordinates = [(x, y) for x in x_coords for y in y_coords]

    # Extract subimages of the specified window size
    half_height = window_size[0] // 2
    half_width = window_size[1] // 2
    subimages_target = []
    coms_target = []
    removed_due_to_size = 0
    removed_due_to_nans = 0

    for coord in coordinates:
        cx, cy = coord
        top = max(cx - half_height, 0)
        bottom = min(cx + half_height, height)
        left = max(cy - half_width, 0)
        right = min(cy + half_width, width)

        subimage = imgdata[top:bottom, left:right]

        if subimage.shape != tuple(window_size):
            removed_due_to_size += 1
        elif np.isnan(subimage).any():
            removed_due_to_nans += 1
        else:
            subimages_target.append(subimage)
            coms_target.append(coord)

    print("Number of subimages removed due to size mismatch:", removed_due_to_size)
    print("Number of subimages removed due to NaNs:", removed_due_to_nans)

    return np.array(subimages_target), np.array(coms_target)



def plt_ground_truth(SBFOdata):
    main = [350, 4000, 350, 4100]
    fig, ax = plt.subplots(nrows=5, ncols=len(SBFOdata), figsize=(4*len(SBFOdata), 4*3), dpi=100)
    for i, k in enumerate(SBFOdata):
        #Image
        ax[0,i].imshow(k['Image'], origin='lower', cmap='gray')
        ax[0,i].set_title(str(k['Index'])+': '+str(k['Composition'])+'%', fontsize = 24, fontweight = "bold")
        ax[0,i].set_axis_off()
        #Px
        ax[1,i].imshow(k['ab_Px'], origin='lower', cmap='coolwarm')
        ax[1,i].set_axis_off()
        #Py
        ax[2,i].imshow(k['ab_Py'], origin='lower', cmap='coolwarm')
        ax[2,i].set_axis_off()

        # Vol (added row to display Vol)
        ax[3, i].imshow(k['Vol'], origin='lower', cmap='coolwarm')
        ax[3, i].set_axis_off()
        Y, X, U, V, cP = plot_polarization_vectors(k)
        ax[4, i].quiver(Y, X, U, V, color=cP, scale=0.1, angles='xy', scale_units='xy', width=0.002)
        ax[4, i].set_title('Ground Truth')
        ax[4, i].set_xlim(main[0], main[1])
        ax[4, i].set_ylim(main[2], main[3])
        ax[4, i].set_aspect('equal') 
        ax[4, i].axis("off")

    plt.tight_layout()
    plt.show()


def pca_plot(imstack_grid, cluster_labels):
    pca = PCA(n_components=2)
    patch_vectors = imstack_grid.reshape(imstack_grid.shape[0], -1) 
    cmap = plt.cm.get_cmap('Set1', len(np.unique(cluster_labels)))
    patch_vectors_2d = pca.fit_transform(patch_vectors)
    plt.scatter(patch_vectors_2d[:, 0], patch_vectors_2d[:, 1], s=5, c = cluster_labels, cmap = cmap )
    plt.title("PCA Projection of Patch Vectors")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()


def k_means_clustering(imstack_grid, n_clusters, com_grid, window_size, step_size):
    patch_vectors = imstack_grid.reshape(imstack_grid.shape[0], -1)
    testing1= KMeans(n_clusters=n_clusters).fit(patch_vectors)

    cluster_labels = testing1.labels_

    # Visualize a few patches from each cluster
    fig, axes = plt.subplots(n_clusters, 3, figsize=(3, 7))
    for cluster in range(n_clusters):
        cluster_patches = imstack_grid[cluster_labels == cluster][:3]  
        for i, patch in enumerate(cluster_patches):
            axes[cluster, i].imshow(patch)
            axes[cluster, i].axis('off')
            axes[cluster, i].set_title(f"Cluster {cluster}")
    plt.tight_layout()
    plt.show()

    for cluster in range(n_clusters):
        print(f"cluster {cluster} size: {len(imstack_grid[cluster_labels == cluster])}")
    
    pca_plot(imstack_grid,cluster_labels )
    plt_clusters(com_grid, window_size, n_clusters, cluster_labels, step_size)


def dbscan_clustering(imstack_grid, eps, min_samples, com_grid, window_size, step_size):
    patch_vectors = imstack_grid.reshape(imstack_grid.shape[0], -1) 
    scaler = MinMaxScaler()
    patch_vectors = scaler.fit_transform(patch_vectors)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(patch_vectors)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])

    if n_clusters == 0:
        print("No clusters found. All points may be noise.")
        return

    fig, axes = plt.subplots(n_clusters, 3, figsize=(9, n_clusters * 3))


    if n_clusters == 1:  # Handle single-cluster case for plotting
        axes = [axes]
    
    for cluster_idx, cluster in enumerate(unique_labels):
        if cluster == -1:  # Skip noise
            continue
        
        cluster_patches = imstack_grid[labels == cluster][:3]  # First 3 patches in this cluster
        for i, patch in enumerate(cluster_patches):
            axes[cluster_idx][i].imshow(patch, cmap='gray')
            axes[cluster_idx][i].axis('off')
            axes[cluster_idx][i].set_title(f"Cluster {cluster}")
    
    plt.tight_layout()
    plt.show()
    
    # Print cluster sizes
    for cluster in unique_labels:
        if cluster == -1:
            print(f"Noise points: {np.sum(labels == -1)}")
        else:
            print(f"Cluster {cluster} size: {np.sum(labels == cluster)}")
    
    # Visualize clusters on the grid
    pca_plot(imstack_grid,labels )
    plt_clusters(com_grid, window_size, n_clusters, labels, step_size)

    
def agglo_clustering(imstack_grid, n_clusters, com_grid, window_size, step_size):
    patch_vectors = imstack_grid.reshape(imstack_grid.shape[0], -1) 
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clustering.fit_predict(patch_vectors)
    pca_plot(imstack_grid,labels )
    plt_clusters(com_grid, window_size, n_clusters, labels, step_size)

def spectral_clustering(imstack_grid, n_clusters, com_grid, window_size, step_size):
    patch_vectors = imstack_grid.reshape(imstack_grid.shape[0], -1) 

    spectral = SpectralClustering(n_clusters=3)
    spectral.fit(patch_vectors)
    labels = spectral.labels_
    pca_plot(imstack_grid,labels )
    plt_clusters(com_grid, window_size, n_clusters, labels, step_size)




def plt_clusters(com_grid, window_size, n_clusters, cluster_labels, step_size):
    chip_width, chip_height = window_size
    cmap = plt.cm.get_cmap('Set1', len(np.unique(cluster_labels)))
    cluster_colors = cmap(cluster_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (center_x, center_y) in enumerate(com_grid):
      bottom_left_x = center_x - chip_width / 2
      bottom_left_y = center_y - chip_height / 2
      rect = patches.Rectangle((bottom_left_x, bottom_left_y), 
                                chip_width, chip_height, 
                                linewidth=1, 
                                edgecolor='none', 
                                facecolor=cluster_colors[i], alpha = .2)
      ax.add_patch(rect)

    ax.set_xlim(com_grid[:, 0].min() - chip_width, com_grid[:, 0].max() + chip_width)
    ax.set_ylim(com_grid[:, 1].min() - chip_height, com_grid[:, 1].max() + chip_height)

    # Add labels
    ax.set_title(f"Clustering Groups, cluster count: {n_clusters}, patch size: {(chip_width, chip_height)}, step size : {step_size}")

    # Add a legend-like colorbar for clusters
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_clusters - 1))
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, label="Cluster ID", ticks=np.arange(0, n_clusters))  # Set ticks
    cbar.set_ticklabels(np.arange(1, n_clusters + 1))  # Set labels as 1 through n_clusters


    # Show the plot
    plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio
    plt.show()