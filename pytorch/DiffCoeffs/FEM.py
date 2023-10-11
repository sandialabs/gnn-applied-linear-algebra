# Graph Neural Networks and Applied Linear Algebra
# 
# Copyright 2023 National Technology and Engineering Solutions of
# Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with
# NTESS, the U.S. Government retains certain rights in this software. 
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: 
# 
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer. 
# 
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution. 
# 
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission. 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
# 
# 
# 
# For questions, comments or contributions contact 
# Chris Siefert, csiefer@sandia.gov 
from sympy import *
import scipy.sparse as sp
import dill
import os
import os.path as osp

x, y, h = symbols('x y h', real=True)
a, b, c, d = symbols('a b c d', real=True)
i, j = symbols('i j', real=True)

class FEM_Builder():
  """
  Builds the problem stiffness matrix for the 2D diffusion problem TODO: put the PDE here where the diffusion is a field given by the matrix
  D = [alpha    0
           0 beta]
  where the alpha and beta can be constant or of the form: TODO: put form here.

  The domain is assumed to be the unit square, meshed into square elements with side-length 1/N where N is the number of grid points.

  The boundary is assumed to be periodic (also viewed as solving on a sphere).

  The necessary algebra is completed via sympy and cached to files for quick retrieval without the need to recompute. Hence it is advantageous to create
  this instance once and re-use it for all the stiffness matrices in the dataset
  """
  def __init__(self, cache_dir='sympy_cache/'):
    # If the cache directory doesn't exist, create it
    self.cache_dir = cache_dir
    if not osp.isdir(cache_dir):
      os.mkdir(cache_dir)

    # If the general stiffness matrix for this type of diffusion has already been cached, load the cache
    if self.K_general_cache_available():
      self.K_general = self.load_K_general_from_cache()
    else:
      self.K_general = self.generate_K_general()
      self.save_K_general_cache()

  def generate_diffusion_field_matrix(self):
    pass

  def generate_convection_field(self):
    return Matrix([[0.0],[0.0]])

  def get_K_general_cache_path(self):
    pass

  def K_general_cache_available(self):
    return osp.isfile(self.get_K_general_cache_path())

  def load_K_general_from_cache(self):
    with open(self.get_K_general_cache_path(), 'rb') as f:
      return dill.load(f)

  def save_K_general_cache(self):
    with open(self.get_K_general_cache_path(), 'wb') as f:
      dill.dump(self.K_general, f)

  def generate_K_general(self):
    # Generate the Diffusion field matrix
    D = self.generate_diffusion_field_matrix()

    # Generate the convection direction
    convection_field = self.generate_convection_field()

    # Generate the basis functions and their gradients
    bases, grads = self.generate_bases_and_grads()

    # Build the general element stiffness matrix - still in terms of i,j, alpha, beta, h
    return self.generate_general_element_stiffness_matrix(D, bases, grads, convection_field)

  def generate_bases_and_grads(self):
    points = [[i*h, j*h], [i*h+h, j*h], [i*h+h, j*h+h], [i*h, j*h+h]]
    B = Matrix([[1, points[i][0], points[i][1], points[i][0]*points[i][1]] for i in range(len(points))])
    phi = a + b*x + c*y + d*x*y
  
    phis = Matrix([[0, 0, 0, 0]])
    for k in range(4):
      coeff = B.LUsolve(eye(4)[:,k])
      phis[k] = phi.subs((p, v) for p,v in zip([a, b, c, d], coeff))
  
    grads = Matrix([[phis[k].diff(z) for z in [x, y]] for k in range(4)]).T
    return phis, grads

  def generate_general_element_stiffness_matrix(self, D, bases, grads, convection_field):
    K = zeros(4,4)
    for k in range(4):
      for l in range(k,4):
        K[k,l] = K[l,k] = integrate((D*grads[:,k] - convection_field*bases[:,k]).dot(grads[:,l]), 
                                    (x, i*h, (i+1)*h), (y, j*h, (j+1)*h))
    return K
  
  def element_to_index_map(self, k, N):
    stencil = {}
    stencil[0] = k

    # For the interior of the domain
    stencil[1] = k+1
    stencil[2] = k+N+1
    stencil[3] = k+N

    # Periodic Boundary Conditions
    if k >= N*(N-1):
      stencil[3] = k-N*(N-1)
      if k == N*N-1: 
        stencil[2] = 0
      else:
        stencil[2] = stencil[3] + 1

    if (k+1)%N == 0:
      stencil[1] = k-(N-1)
      if k != N*N-1:    # j == N*N-1 handled above
        stencil[2] = stencil[1] + N

    return stencil

  def assemble_problem_stiffness_matrix_from_element_stiffness(self, K, N):
    row = []
    col = []
    data = []
    def add_entry(i,j,v):
      row.append(i)
      col.append(j)
      data.append(v)

  # Loop through the elements - addressing by their lower-left corner
    for elem_idx in range(N*N):
      i_idx = elem_idx % N
      j_idx = elem_idx // N
      # Get the specifc element stiffness matrix
      K_elem = K(i_idx,j_idx)

      # Get the element_to_index_map
      e_to_i = self.element_to_index_map(elem_idx,N)

      for v in range(4):
        # Contributions from vertices
        add_entry(e_to_i[v],         e_to_i[v],         K_elem[v,v])
        # Contributions from edges
        add_entry(e_to_i[v],         e_to_i[(v+1) % 4], K_elem[v,(v+1) % 4])
        add_entry(e_to_i[(v+1) % 4], e_to_i[v],         K_elem[v,(v+1) % 4])
        # Contributions from corner
        add_entry(e_to_i[v],         e_to_i[(v+2) % 4], K_elem[v,(v+2) % 4])

    return row, col, data
   
class ConstantDiffusionFEM_Builder(FEM_Builder):
  def __init__(self):
    self.alpha, self.beta = symbols('alpha beta', real=True)
    super().__init__()

  def generate_diffusion_field_matrix(self):
    return Matrix([[self.alpha, 0], [0, self.beta]])

  def get_K_general_cache_path(self):
    return self.cache_dir + 'general_K_constant_diffusion.sympy'

  def generate_problem_stiffness_matrix(self, prob_alpha, prob_beta, N):
    K_problem = self.K_general.subs({self.alpha: prob_alpha, self.beta: prob_beta, h:1.0/N})
    K_problem = lambdify((i,j), K_problem)
    return self.assemble_problem_stiffness_matrix_from_element_stiffness(K_problem, N)

class CosineDiffusionFEM_Builder(FEM_Builder):
  def __init__(self):
    self.alpha = Function('alpha', real=True)(x,y)
    self.beta = Function('beta', real=True)(x,y)
    self.theta_alpha_x = symbols('theta_x^alpha', real=True)
    self.theta_alpha_y = symbols('theta_y^alpha', real=True)
    self.theta_beta_x  = symbols('theta_x^beta',  real=True)
    self.theta_beta_y  = symbols('theta_y^beta',  real=True)
    self.test_alpha    = cos(self.theta_alpha_x*2*pi*x)**2*cos(self.theta_alpha_y*2*pi*y)**2 + 0.1
    self.test_beta     = cos(self.theta_beta_x*2*pi*x)**2*cos(self.theta_beta_y*2*pi*y)**2 + 0.1   
    super().__init__()

  def generate_diffusion_field_matrix(self):
    return Matrix([[self.alpha, 0], [0, self.beta]])
     
  def get_K_general_cache_path(self):
    return self.cache_dir + 'general_K_cosine_diffusion.sympy'

  def generate_general_element_stiffness_matrix(self, D, bases, grads, convection_field):
    K_general = super().generate_general_element_stiffness_matrix(D, bases, grads, convection_field)
    return K_general.subs({self.alpha: self.test_alpha, self.beta:self.test_beta}).doit()

  def generate_problem_stiffness_matrix(self, theta_alpha_x, theta_alpha_y, theta_beta_x, theta_beta_y, N):
    K_problem = self.K_general.subs({self.theta_alpha_x: theta_alpha_x, 
                                     self.theta_alpha_y: theta_alpha_y,
                                     self.theta_beta_x: theta_beta_x,
                                     self.theta_beta_y: theta_beta_y,
                                     h:1.0/N}).doit()
    K_problem = lambdify((i,j), K_problem)
    return self.assemble_problem_stiffness_matrix_from_element_stiffness(K_problem, N)

class CosineDiffusionConvectionFEM_Builder(CosineDiffusionFEM_Builder):
    def __init__(self):
      super().__init__()

    def generate_convection_field(self):
      return 0.1*Matrix([[1], [0]])

    def get_K_general_cache_path(self):
      return self.cache_dir + 'general_K_cosine_diffusion_convection.sympy'
    
        
if __name__ == '__main__':
  import sys
  import numpy as np
  import scipy.sparse as sp

  init_printing()
  np.set_printoptions(threshold=sys.maxsize, linewidth=200)

  prob_alpha = 1
  prob_beta = 0.01
  N = 10 

  FEM = ConstantDiffusionFEM_Builder()
  row, col, data = FEM.generate_problem_stiffness_matrix(prob_alpha, prob_beta, N)
  A_Constant = sp.coo_matrix((data, (row, col)), shape=(N*N,N*N)).toarray()
  # print(A_Constant)

  thetas = [1, 1, 1, 1]
  FEM = CosineDiffusionFEM_Builder()
  row, col, data = FEM.generate_problem_stiffness_matrix(*thetas, N)
  A_CosineDiffusion = sp.coo_matrix((data, (row, col)), shape=(N*N,N*N)).toarray()
  # print(A_CosineDiffusion)

  thetas = [1, 1, 1, 1]
  FEM = CosineDiffusionConvectionFEM_Builder()
  row, col, data = FEM.generate_problem_stiffness_matrix(*thetas, N)
  A_CosineDiffusionConvection = sp.coo_matrix((data, (row, col)), shape=(N*N,N*N)).toarray()
  print(A_CosineDiffusionConvection)