import numpy as np
import scipy.spatial as spatial
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs

def find_dt(a,gamma):    
    dt = np.sqrt(2*gamma/a)
    return dt

def calculate_gravitational_force(r,m,i,j):
    r1 = r[i,:]
    r2 = r[j,:]
    m1 = m[i]
    m2 = m[j]
    r = r2-r1
    distance = np.linalg.norm(r,axis = 1)
    f = np.divide(m1*m2[:,0],distance**2)
    r_norm = np.divide(r,np.reshape(distance,(distance.shape[0],1)))
    force_vector = np.multiply(r_norm,np.reshape(f,(f.shape[0],1)))
    sum_force = np.sum(force_vector,axis = 0)
    return sum_force    

def gravity(r,m,N):
    if r.shape[0]>N:
        nbrs = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(r)
    else:
        nbrs = NearestNeighbors(n_neighbors=r.shape[0], algorithm='ball_tree').fit(r)
    distances, indices = nbrs.kneighbors(r)
    indices = indices[:,1:]
    force = []
    for i,j in enumerate(indices):
        force.append(calculate_gravitational_force(r,m,i,j))
    force = np.array(force)
    return force
        
def replace_nearest_points(r,m,index):
    all_idx = set(np.arange(0,r.shape[0]))
    not_index = list(all_idx.difference(set(np.hstack(index))))
    replaced_mass = []
    replaced_r = []
    for j,i in enumerate(index):
        replaced_mass.append(np.sum(m[i]))
        replaced_r.append(np.sum(np.multiply(m[i],r[i,:]),axis = 0)/np.sum(m[i]))    
    replaced_r = np.array(replaced_r)    
    replaced_mass = np.array(replaced_mass)
    replaced_mass = np.reshape(replaced_mass,(replaced_mass.shape[0],1))
    r = np.concatenate((r[not_index,:],replaced_r))
    m = np.concatenate((m[not_index],replaced_mass))
    return r,m  

def find_nearest_points(r,epsilon):
    point_tree = spatial.cKDTree(r)
    neighbors = point_tree.query_ball_point(r,epsilon,p=2.0)
    for i,j in enumerate(neighbors):
        if len(j)==1:
            neighbors[i] = []       
    index = []
    used_points =  [False for i in range(len(neighbors))]
    for i,j in enumerate(neighbors):
        if not used_points[i]:
            seed = j
            if seed:
                size_one = 0
                size_two = len(seed)
                while size_two != size_one:
                    size_one = len(seed)
                    idx = neighbors[seed]
                    idx = np.unique(np.hstack(idx))
                    seed = idx
                    size_two = len(seed)
                for i in idx:
                    used_points[i]=True
                index.append(seed)
    return index

number_of_particles = 1000
epsilon = 0.02

# number_of_particles = int(input('Enter Number of Particles:'))
# epsilon = float(input('Enter Epsilon Value:'))

r = np.random.random((number_of_particles,2))

# r, true_labels = make_blobs(n_samples=200,centers=3,cluster_std=0.8)
# nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(r)
# distances, indices = nbrs.kneighbors(r)
# distances = distances[:,1]
# epsilon = np.percentile(distances,50)

m = np.ones((r.shape[0],1))
gamma = epsilon/2

fig = plt.figure()
ax = fig.add_subplot(111)

while r.shape[0]>1:    
    index = find_nearest_points(r,epsilon)
    
    if index:
        [r,m] = replace_nearest_points(r,m,index)
        
    if r.shape[0]>1:
        f = gravity(r,m,r.shape[0])
        a = np.divide(f,m)
        a_norm = np.linalg.norm(a,axis = 1)
        dt = find_dt(np.max(a_norm),gamma)
        r = 0.5*a*dt**2+r    
    plt.cla()
    plt.scatter(r[:,0],r[:,1],s = m,c = m, cmap = 'PRGn')       
    plt.pause(0.0000001)
    plt.xlim((0,1))
    plt.ylim((0,1))
    ax.set_aspect('equal')
    fig.canvas.draw()
    fig.canvas.flush_events()
    
plt.show()