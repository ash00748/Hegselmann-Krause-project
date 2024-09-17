import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from numba import njit

@njit()
def HK_2D_step(pos_arr, R):
    '''
    Returns new potion of particles based on neighbors in circle of radius R
    '''
    N = len(pos_arr)
    pos_new = np.empty((N, 2))
    for i in range(N):
        res_pos = np.array([0.0, 0.0])
        count = 0
        for j in range(N):
            if (np.linalg.norm(pos_arr[i] - pos_arr[j]) <= R):
                res_pos += pos_arr[j]
                count += 1

        pos_new[i] = res_pos / count
    return pos_new

njit()
def HK_2D_sim(pos_init, R, eps, max_iter):
    '''
    Simulates HK model for 2D particles
    '''
    pos = pos_init.copy()
    pos_t_array = np.empty((0, pos.shape[0], pos.shape[1]), dtype=np.float64)
    pos_t_array = np.append(pos_t_array, np.array([pos.copy()]), axis=0)
    for i in range(max_iter):
        pos_new = HK_2D_step(pos, R)
        if np.linalg.norm(pos_new - pos) < eps:
            break
        pos = pos_new
        pos_t_array = np.append(pos_t_array, np.array([pos.copy()]), axis=0)
    return pos_t_array, i != max_iter - 1

@njit()
def caln_n_clusters(positions, R):
    N = positions.shape[0]
    cluster_of_agent = np.zeros(N, dtype=np.int64)
    cur_claster = 1
    for i in range(N):
        if cluster_of_agent[i] == 0:
            cluster_of_agent[i] = cur_claster
            for j in range(i+1, N):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= R:
                    cluster_of_agent[j] = cur_claster
            cur_claster += 1

    cluster_of_agent -= 1
    n_clusters = cur_claster - 1
    cluster_sizes = np.zeros(n_clusters, dtype=np.int64)
    for i in range(N):
        cluster_sizes[cluster_of_agent[i]] += 1

    return cluster_of_agent, cluster_sizes

def draw_HK_2D_simulation(res_arr, cmap=None):
    plt.figure()
    N = res_arr.shape[1]
    steps = res_arr.shape[0]
    plt.gca().set_aspect('equal')
    if cmap:
        cmap = mp.cm.get_cmap(cmap)
        norm=plt.Normalize(vmin=0, vmax=N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    
    for i in range(N):
        if cmap:   
            color=np.array(cmap(norm(i)))*0.9
        else:
            color = np.random.random(3)*0.7

        # plt.plot(res_arr[:, i, 0], res_arr[:, i, 1], '->', alpha = 0.5, color = color )
        for t in range (steps-1):
            plt.arrow(res_arr[t,i,0], res_arr[t,i,1], res_arr[t+1,i,0]-res_arr[t,i,0], res_arr[t+1,i,1]-res_arr[t,i,1],
                      head_width=0.01, head_length=0.01, fc=color, ec=color, alpha=0.5)

    plt.scatter(res_arr[-1, :, 0], res_arr[-1, :, 1], label = 'final clusters')
    plt.ylabel('y opinion')
    plt.xlabel('x opinion')
    plt.legend(loc="upper right")
    # plt.grid(True)
    plt.show()