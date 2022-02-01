import numpy
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = cm.Spectral
import numpy as np
from scipy.sparse import spdiags, diags
from tqdm import tqdm
import copy

from scipy.linalg import solve_banded

L=300; #Lenght of system
J=L*1;I=J #Number of gridpoints

# D=[0.125,0.5] #Diffusion parameters for Turing system
D=[1,40,1] #Diffusion parameters for Schnakenberg system

dx = float(L)/float(J-1)
x_grid = numpy.array([j*dx for j in range(J)])

T = 299 #Total time
N=T*10 #Number of timepoints
dt = float(T)/float(N-1)
t_grid = numpy.array([n*dt for n in range(N)])

n_species=3 #number of chemical species/variables/equations

#Define initial concentrations of chemical system.
#In this case, a uniform concentration of 0.1 (with some noise) is defined through space for both chemical species.
U0 = []
perturbation=0.001
steadystates=[0.1]*n_species
np.random.seed(1)
initialJ=1 #size of initial system
for index in range(n_species):
    U0.append(np.random.uniform(low=steadystates[index] - perturbation, high=steadystates[index] + perturbation, size=(initialJ)))

alpha = [D[n]*dt/(2.*dx*dx) for n in range(n_species)]

def A(alphan,J):
    bottomdiag = [-alphan for j in range(J-1)]
    centraldiag = [1.+alphan]+[1.+2.*alphan for j in range(J-2)]+[1.+alphan]
    topdiag = [-alphan for j in range(J-1)]
    diagonals = [bottomdiag,centraldiag,topdiag]
    A = diags(diagonals, [ -1, 0,1]).toarray()
    return A

def B(alphan,J):
    bottomdiag = [alphan for j in range(J-1)]
    centraldiag = [1.-alphan]+[1.-2.*alphan for j in range(J-2)]+[1.-alphan]
    topdiag = [alphan for j in range(J-1)]
    diagonals = [bottomdiag,centraldiag,topdiag]
    B = diags(diagonals, [ -1, 0,1]).toarray()
    return B


# def schnakenberg(u,c=[0.1,1,0.9,1]):
#     c1,cm1,c2,c3 = c
#     f_u0 = c1 - cm1*u[0] + c3*(u[0]**2)*u[1]
#     f_u1 = c2 - c3*(u[0]**2)*u[1]
#     return f_u0,f_u1

def turing_3N(u):
    f_a = 5*u[0] - 6*u[1] + 3*u[2] + 1
    f_b = 6*u[0] - 7*u[1] - 3*u[2] + 1
    f_c = -3*u[0] + 4*u[1] + 1*u[2] + 1
    return f_a, f_b, f_c

#standard function for handling topolgy and parameter
def interaction(u_array,topology,parameter,constant = 0):
    interaction_matrices = topology * parameter
    f_inter = np.dot(interaction_matrices,u_array) + constant
    return f_inter

U = copy.deepcopy(U0)

currentJ = initialJ
U_record = []
for species_index in range(n_species):
    U_record.append(np.zeros([J, T]))  # DO NOT SIMPLIFY TO U_record = [np.zeros([J, I, T])]*n_species

# These two lists contain the A and B matrices for every chemical specie. They are adapted to the size of the field,
# meaning that if the field is J=3, the matrix will be 3x3.
A_list = [A(alphan, currentJ) for alphan in alpha]
B_list = [B(alphan, currentJ) for alphan in alpha]

# for loop iterates over time recalculating the chemical concentrations at each timepoint (ti).
currentJ = initialJ
for ti in tqdm(range(N), disable=False):
    U_new = copy.deepcopy(U)
    f0 = turing_3N(U)

    diff = []
    # iterate over every chemical specie when calculating concentrations.
    for n in range(n_species):
        U_new[n] = numpy.linalg.solve(A_list[n], B_list[n].dot(U[n]) + f0[n] * (dt / 2))

    hour = ti / (N / T)

    if hour % 1 == 0:  # only grow and record at unit time (hour)
        for n in range(n_species):
            U_new[n] = np.concatenate((U_new[n], [U_new[n][-1]]))
            # System grows by one gridpoint. New gridpoint obtains concentration value of neighbour.
            U_record[n][:, int(hour)] = np.concatenate((U_new[n], np.zeros(J - currentJ - 1)))
            # Solution added into array which records the solution over time (JxT dimensional array)

        currentJ += 1  # System size grows by one

        # A and B must be updated as system size increases. They are calculated with the new J (currentJ)
        A_list = [A(alphan, currentJ) for alphan in alpha]
        B_list = [B(alphan, currentJ) for alphan in alpha]

    U = copy.deepcopy(U_new)

reduced_t_grid = np.linspace(0,T,T)
def surfpattern(results,grids=[x_grid,reduced_t_grid],morphogen = 0):
    results = np.transpose(results[morphogen])
    x_grid = grids[0]
    t_grid = grids[1]
    values = results.reshape(len(t_grid),len(x_grid))
    x, t = np.meshgrid(x_grid, t_grid)
    plt.contourf(t,x,results, cmap=cmap)
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.show()

surfpattern(U_record)