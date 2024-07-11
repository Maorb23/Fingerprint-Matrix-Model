#!/usr/bin/env python
# coding: utf-8

# # INDIVIDUAL MATRIX MODEL

# In this fast-paced world, we frequently interact with people from different backgrounds who hold diverse beliefs and morals. 
# 
# Moreover, personal appearance varies for each individual.
# 
# We believe that each person has a unique marker that determines how they present themselves to the outside world. 
# 
# Furthermore, we believe that in each interaction, we can **quantify** the **individual identity** of each person and the **shared stimulus** between all people.
# 
# We suggest that each individual will have a **weight matrix**, which is multiplied by the **universal stimulus matrix** (which is the same for all subjects). 
# 
# The multiplication of the **weight matrix** and the **stimulus matrix** will result in the **response matrix** of the individual.
# 
# Let's model it to make it clearer:

# We have our underlining assumptions:
# 
# 1. i = 1,2...,m for m **subjects**
# 2. We have our index v for different **voxels** and our index d for different **timeframes**
# 3. We'll define our matrix $X_i$ with dimensions $v_{rows}$ X $d_{columns}$
# 
# Let's write our suggested model:
# 
# For the matrices:  
# 
# $X_i \in \mathbb{R}^{v x d}$, $W_i \in \mathbb{R}^{v x k}$, $S \in \mathbb{R}^{k x d}$ : 
# 
# Model_1 is:
# 
# $$
# X_i = W_i \cdot S + E_i \quad, \quad \text{ for i subject in our sample}
# $$
# 
# Note: to ensure uniqueness of coordinates it is necessary that $W_i$ has
# linearly independent columns.
# 
# Thus, in our model we make a bigger assumtion that the weights matrix is **orthogonal**, meaning: $\quad W_i^T W_i = I _{k}$
# 
# 
# Now we're going to optimize our $W_i$ in the following method:
# 
# 
# 
#     

# 1. Our goal is to **minimize**:  
# 
# $$
# Min_{W_i,S} \quad \sum_{i = 1}^{m} \| X_{i} - W_{i} S \|_F^2
# $$
# 
# 2. Select initial $W_i$
# <br>
# 
# 3. Set $S = \frac{1}{m} \sum_{i = 1}^{m} W_i^T X_i$
# <br>
# 
# 4. We have m separate subproblems of the form $ \quad \sum_{i = 1}^{m} \| X_{i} - W_{i} S \|_F^2$
# 
# <br>
# 
# 5. Our solution for each subject in each iteration is:
# 
# $$
# W_{i} = \tilde U_{i} \tilde V_{i}^T \quad \text{where} \quad \tilde U_{i} \tilde D_{i} \tilde V_{i}^T = SVD(X_{i}S^T)
# $$
# 
# One question immediately comes to mind: 
# 
# ***WHAT'S SVD?***
# 
# 
# <br>

# ## SVD 
# 
# SVD says we can write **ANY** matrix A as $A_{mxn} = U_{mxm} D_{mxn} V^T_{nxn} \quad$ where:
# 
# **U** is an **orthogonal matrix** (each of its rows and columns are orthonormal vectors). in general, if A is an m x n matrix, then V is an m x m matrix:
# 
# <br>
# 
# $$
# U = \begin{bmatrix} | & | & | & | \\ u_{1} & u_{2} & ... & u_{n} \\ | & | & | & | \end{bmatrix}
# $$
# 
# <br>
# 
# **D** is matrix with the **singular values** on the diagonal and zeros elsewhere. In general if A is an m x n matrix, then D is an m x n matrix. We have in our main diagonal the singular values of A which equal to the square root of the eigenvalues of $A^TA$ or $AA^T$. The rest of the elements are zeros:
# 
# <br>
# 
# $$
# D = \begin{bmatrix} d_{1} & 0 & 0 & 0 & 0 &... &0 \\ 0 & d_{2} & 0 & 0 & 0 &... &0 \\ 0 & 0 & ... & 0 & 0 &... &0 \\ 0 & 0 & 0 & d_{n} & 0 &... &0 \end{bmatrix}
# $$
# 
# <br>
# 
# **V** is an **orthogonal** matrix too. In general, if A is an m x n matrix, then V is an n x n matrix.
# 
# <br>
# 
# $$
# V = \begin{bmatrix} | & | & | & | \\ v_{1} & v_{2} & ... & v_{n} \\ | & | & | & | \end{bmatrix}
# $$

# ### Let's visualise it shall we?

# In[99]:


import numpy as np
import matplotlib.pyplot as plt

def plot_circle(ax, center, radius, **kwargs):
    """Plot a circle with a given center and radius."""
    circle = plt.Circle(center, radius, **kwargs)
    ax.add_artist(circle)

# Generate a circle of points
theta = np.linspace(0, 2 * np.pi, 100)
circle = np.vstack((np.cos(theta), np.sin(theta)))

# Define a matrix to be decomposed
A = np.array([[3, 1], [1, 2]])

# Perform SVD
U, Sigma, Vt = np.linalg.svd(A)
Sigma_matrix = np.diag(Sigma)

# Apply transformations
circle_U = U @ circle
circle_SigmaU = Sigma_matrix @ circle_U
circle_SigmaUVt = Vt @ circle_SigmaU
circle_A = A @ circle

# Plotting
fig, ax = plt.subplots(1, 5, figsize=(25, 5))

# Original circle
ax[0].plot(circle[0, :], circle[1, :], 'b')
plot_circle(ax[0], (0, 0), 1, color='b', fill=False)
ax[0].set_xlim(-4, 4)
ax[0].set_ylim(-4, 4)
ax[0].set_aspect('equal', 'box')
ax[0].set_title("Original Circle")

# Transformed by U
ax[1].plot(circle_U[0, :], circle_U[1, :], 'm')
plot_circle(ax[1], (0, 0), 1, color='m', fill=False)
ax[1].set_xlim(-4, 4)
ax[1].set_ylim(-4, 4)
ax[1].set_aspect('equal', 'box')
ax[1].set_title("Transformation by $U$")
ax[1].annotate(f"$\\sigma_1 = {Sigma[0]:.2f}$", (0.5, 0.5), color='black', fontsize=12, ha='center')
ax[1].annotate(f"$\\sigma_2 = {Sigma[1]:.2f}$", (-0.5, -0.5), color='black', fontsize=12, ha='center')

# Transformed by Sigma
ax[2].plot(circle_SigmaU[0, :], circle_SigmaU[1, :], 'g')
plot_circle(ax[2], (0, 0), 1, color='g', fill=False)
ax[2].set_xlim(-4, 4)
ax[2].set_ylim(-4, 4)
ax[2].set_aspect('equal', 'box')
ax[2].set_title("Transformation by $\Sigma$")
ax[2].annotate(f"$\\sigma_1 = {Sigma[0]:.2f}$", (Sigma[0]/2, 0), color='black', fontsize=12, ha='center')
ax[2].annotate(f"$\\sigma_2 = {Sigma[1]:.2f}$", (0, Sigma[1]/2), color='black', fontsize=12, ha='center')

# Transformed by Vt
ax[3].plot(circle_SigmaUVt[0, :], circle_SigmaUVt[1, :], 'r')
plot_circle(ax[3], (0, 0), 1, color='r', fill=False)
ax[3].set_xlim(-4, 4)
ax[3].set_ylim(-4, 4)
ax[3].set_aspect('equal', 'box')
ax[3].set_title("Transformation by $V^T$")
ax[3].annotate(f"$\\sigma_1 = {Sigma[0]:.2f}$", (Sigma[0]/2, 0), color='black', fontsize=12, ha='center')
ax[3].annotate(f"$\\sigma_2 = {Sigma[1]:.2f}$", (0, Sigma[1]/2), color='black', fontsize=12, ha='center')

# Transformation by A directly
ax[4].plot(circle_A[0, :], circle_A[1, :], 'c')
plot_circle(ax[4], (0, 0), 1, color='c', fill=False)
ax[4].set_xlim(-4, 4)
ax[4].set_ylim(-4, 4)
ax[4].set_aspect('equal', 'box')
ax[4].set_title("Transformation by $A$")
ax[4].annotate(f"$\\sigma_1 = {Sigma[0]:.2f}$", (Sigma[0]/2, 0), color='black', fontsize=12, ha='center')
ax[4].annotate(f"$\\sigma_2 = {Sigma[1]:.2f}$", (0, Sigma[1]/2), color='black', fontsize=12, ha='center')

plt.show()


# Now, that's beautiful! 
# 
# As we can see (carefully) U and $V^T$ rotate the circle and $\Sigma$ stretches it. We get the exact same operation as if we simply multiplied by A!!

# ## Imports

# In[61]:


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib as plt
import os


# ## Reading all the files from my folders:

# In[62]:


directory = r"C:\Users\maorb\CSVs"

# Our columns of interest
columns = [
    'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
    'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
    'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
]

# Function to read files and process them
def read_and_process_file(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        return None
    
    df.columns = df.columns.str.strip() # Strip whitespace from column names
    return df[columns]

# Get all files that start with "Argaman"
files = [f for f in os.listdir(directory) if f.startswith("Argaman") and (f.endswith(".csv") or f.endswith(".xlsx"))]

# Read and process all files
dataframes = [read_and_process_file(os.path.join(directory, file)) for file in files]

# Drop any None values in case some files were not processed
dataframes = [df for df in dataframes if df is not None]

# Find the minimum length of all dataframes
min_length = min(len(df) for df in dataframes)

# Trim all dataframes to the minimum length
dataframes = [df.iloc[:min_length, :] for df in dataframes]

# Combine the dataframes
combined_data = dataframes[0]
for df in dataframes[1:]:
    df = df.add_suffix(f'_{df}')  # Add suffix to each dataframe to avoid column name clashes
    combined_data = combined_data.join(df)



# In[64]:


dataframes[0]


# Note: In our datasets we have 
# 
# $$
# Rows_{timeframes}\quad x \quad Columns_{muscles},\text{ or }  dxv
# $$
# 
# <br>
# 
# Thus, we're going to transpose the matrix!

# In[65]:


dataframes_Trans = [df.T for df in dataframes]


# ## Our Identity Matrix model

# In[86]:


def W_i_calc3(X_i, S):
    X_S_T = np.dot(X_i, S.T)
    U_i, Sigma, V_i_T = np.linalg.svd(X_S_T, full_matrices=False)
    W_i = np.dot(U_i, V_i_T)
    return W_i

def SRM(X, tol=1e-10, max_iter=100000):
    dist_vec = []
    indices = []
    W_i_vec = []
    W_i_new_vec = []
    n = 16
    m = 8
    W_i_new_group = np.ones((m, n, n))
    W_i = np.ones((n,n)) # Initialize W_i as a matrix of ones
    iter_count = 0
    converged = False
    k = 1
    while not converged and iter_count < max_iter:
            for j, X_i in enumerate(X):
                # Compute S for the current j
                S = (1 / len(X)) * sum(np.dot(W_i.T, X_i) for W_i in W_i_new_group)
                W_i_new_group[j] = W_i_calc3(X_i, S)

                # Calculate distance for convergence check
                dist = np.linalg.norm(X_i - np.dot(W_i_new_group[j], S), 'fro')
                dist_vec.append(dist)
                indices.append(k)
                k += 1

                if dist < tol:
                    converged = True
                    break

            iter_count += 1
            if iter_count >= max_iter:
                converged = True

        # We can find the argmin W_i of the function ||X - W_i @ S||^2 by finding the argmin of ||X - U_i @ D_i @ V_i^T @ S||^2 :
    A = (np.linalg.norm(X_i - np.dot(W_i[j], S), 'fro'))**2  # We'll find the argmin W_i of this function

    return iter_count, W_i_new_group, S, A, dist_vec, indices


# In[3]:


len(dataframes_Trans[0])


# In[87]:


iter_count, W_i_new_group, S, A, dist_vec, indices = SRM(dataframes_Trans)


# Note: We did it with small number of iterations:

# In[94]:


pd.DataFrame({'Loops entered': iter_count, 'Adjustments': len(indices)}, index=['SRM'])


# #### W_i for example:

# In[95]:


W_i_exam = pd.DataFrame(W_i_new_group[0])
W_i_exam


# Indeed, we have 16X16 matrix, row and matrix for each different muscle.

# ## Let's plot the distances

# In[96]:


import matplotlib.pyplot as plt
# Create a scatter plot using seaborn
dists = pd.DataFrame({'indices': indices, 'Distances': dist_vec})
sns.scatterplot(dists, x= 'indices', y='Distances', )

# Set the labels and title using seaborn's functionality
sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")
#sns.label(x = "Indices")


# with tol=1e-10, max_iter=100000 and 59 adjustments
