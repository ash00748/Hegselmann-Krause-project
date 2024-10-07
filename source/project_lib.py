import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from numba import njit, prange
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from IPython import display
from numba.typed import List


@njit(parallel=True)
def HK_2D_step_fast(pos_arr, R):
    '''
    Returns new potion of particles based on neighbors in circle of radius R
    '''

    cell_size = 1.01 * R # 
    n_cells = int(1 + np.ceil(1.0 / cell_size))
    grid_side = n_cells + 2
    N = len(pos_arr)
    grid = np.empty((grid_side, grid_side, N+1), dtype=np.int32) # padding around the grid

    for x in prange(grid_side):
        for y in range(grid_side):
            grid[x, y, 0] = 0

    point2grid = np.empty((N, 2), dtype=np.int32)
    for i in range(N):
        x = 1 + int(pos_arr[i, 0] / cell_size)
        y = 1 + int(pos_arr[i, 1] / cell_size)
        grid[x, y, 0] += 1
        grid[x, y, grid[x, y, 0]] = i
        point2grid[i, 0] = x
        point2grid[i, 1] = y

    pos_new = np.empty((N, 2))
    for i in prange(N):
        res_pos = np.array([0.0, 0.0])
        count = 0
        x = point2grid[i, 0]
        y = point2grid[i, 1]
        for ix in [-1, 0, 1]:
            for iy in [-1, 0, 1]:
                for k in range(1, grid[x+ix, y+iy, 0] + 1):
                    j = grid[x+ix, y+iy, k]
                    if (np.linalg.norm(pos_arr[i] - pos_arr[j]) <= R):
                        res_pos += pos_arr[j]
                        count += 1
                        
        pos_new[i] = res_pos / count

    return pos_new

# @njit()
def HK_2D_sim_fast(pos_init, R, eps, max_iter):
    '''
    Simulates HK model for 2D particles
    '''
    pos = pos_init.copy()
    pos_t_array = np.empty((0, pos.shape[0], pos.shape[1]), dtype=np.float64)
    pos_t_array = np.append(pos_t_array, np.array([pos.copy()]), axis=0)
    for i in range(max_iter):
        pos_new = HK_2D_step_fast(pos, R)
        if np.linalg.norm(pos_new - pos) < eps:
            break
        pos = pos_new
        pos_t_array = np.append(pos_t_array, np.array([pos.copy()]), axis=0)
    return pos_t_array, i != max_iter - 1
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

# @njit()
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

def draw_HK_2D_simulation(res_arr, N=None, R=None, text=None, cmap=None):
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
    plt.title(f"N:{N} eps:{R} {text}")
    # plt.grid(True)
    plt.show()

def HK_2D_display_steps(res_arr):
    for res in res_arr:
        plt.scatter(res[:, 0], res[:, 1])
        plt.ylim(-0.1, 1)
        plt.xlim(-0.1, 1)
        plt.show()
    
def animate_HK_2D_simulation(res_arr, file_name='scatter.gif'):
    Figure = plt.figure()

    plt.gca().set_aspect('equal')
    # creating a plot
    scat = plt.scatter(res_arr[0, :, 0], res_arr[0, :, 1]) 
    
    plt.xlim(-0.1,1.0)  
    plt.ylim(-0.1,1.0)   

    def AnimationFunction(frame):
        offsets = []
        scat.set_offsets([x for x in res_arr[frame]])
        return scat,

    anim_created = FuncAnimation(Figure, AnimationFunction, frames=res_arr.shape[0])

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=5,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
    anim_created.save(file_name, writer=writer)

def compare(x, y, eps2):
    '''
    Функция, которая сравнивает текущую и предыдущую итерацию
    
    Parameters
    ----------
    x: ndarray, shape (n,)
        Профиль мнений на текущей итерации, состоящий из n агентов
    y: ndarray, shape (n,)
        Профиль мнений на предыдущей итерации, состоящий из n агентов
    eps2: float
        Допустимая точность
        
    Returns
    -------
    result: bool
        True, если произошла "заморозка" модели
        False, если "заморозки" не произошло
    '''
    result = 1
    for k in range(len(x)):
        if abs(x[k] - y[k]) >= eps2:
            result = 0
    result = bool(result)
    return result


def order_parameter(x, eps, target_ind=-1):
    summary = 0
    n = len(x)
    for i in range(n):
        if i == target_ind:
            continue
        for j in range(n):
            if j != target_ind:
                if abs(x[i]-x[j]) < eps:
                    summary += 1
                else:
                    continue
    if target_ind == -1:
        return summary/(n**2)
    else:
        return summary/((n-1)**2)

def new_op(x, i, eps):
    count = 1
    summ = x[i]
    for k in range(len(x)):
        if (abs(x[i] - x[k]) < eps) and (i != k):
            # print(i, j, x[i], x[j])
            summ += x[k]
            count += 1
    # print(summ, count)
    return summ/count, count

def is_consensus(x, fix):
    for i in range(1, fix):
        if x[i-1] != x[i]:
            return False
    if x[fix - 1] != x[fix + 1]:
        return False
    for i in range(fix + 2, len(x)):
        if x[i-1] != x[i]:
            return False
    return True