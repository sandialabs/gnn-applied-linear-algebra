% test_jacobi example

% Graph Neural Networks and Applied Linear Algebra
% 
% Copyright 2023 National Technology and Engineering Solutions of
% Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with
% NTESS, the U.S. Government retains certain rights in this software. 
% 
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met: 
% 
% 1. Redistributions of source code must retain the above copyright
% notice, this list of conditions and the following disclaimer. 
% 
% 2. Redistributions in binary form must reproduce the above copyright
% notice, this list of conditions and the following disclaimer in the
% documentation and/or other materials provided with the distribution. 
% 
% 3. Neither the name of the copyright holder nor the names of its
% contributors may be used to endorse or promote products derived from
% this software without specific prior written permission. 
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
% “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
% A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
% HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
% SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
% LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
% DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
% THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
% 
% 
% 
% For questions, comments or contributions contact 
% Chris Siefert, csiefer@sandia.gov 


function test_jacobi(nx)

A=laplacianfun([nx,nx]);
N=size(A,1);
D=spdiags(diag(A),0,N,N);
omega=2/3; % for giggles
[II,JJ,VV]=find(A);


rng(24601,'twister');
x0_exact=rand(N,1);
b_exact=rand(N,1);


%% Edge features %%
% 1 immutable: A_ij
e_attr_fixed = VV;
% 1 mutable: 0 input, Aij xj on output
e_attr_mutable = 0*VV;

%% Vertex features
% 2 immutable: A_ii, b_i
v_attr_fixed = [diag(A), b_exact];
% 1 mutable: x_i input x_i,  + omega D_i^{-1}(bi - \sum_j A_ij xj) on output
v_attr_mutable = x0_exact;

%% Global features
% 1 immutable: omega
g_attr_fixed = omega;
% 0 mutable


x_fixed={v_attr_fixed,e_attr_fixed,g_attr_fixed};
x_mutable={v_attr_mutable,e_attr_mutable,[]};

output = gnn(N,II,JJ,x_fixed,x_mutable,...
             @vertex_update,@edge_update,[],...
             1,@agg_edge_to_vertex,[],[],[]);

% Result vector
x1_nn=output{1};

% Jacobi
%x_k+1 = x_k + w D^{-1} ( b-A x_k)
x1_exact = x0_exact + omega * (D \ (b_exact - A * x0_exact));


fprintf('Rel Error = %22.16e\n',norm(x1_nn - x1_exact) / norm(x1_exact));


%% GNN Operation Ordering:
% c_ij = edge_update(e_attr,vattr_i,vattr_j,g,model_params)
% cbar_i = agg_edge_to_vertex(z_ij)
% v_attr = vertex_update(v_attr,cbar_i,g,model_params)
% e_aggregated = agg_edge_to_global(z_ij)
% v_aggregated = agg_vertex_to_global(v_attr)
% g  = global_update(g,v_aggregated,e_aggregated,model_params)


function out_c_ij = edge_update(e_attr,vattr_i,vattr_j,g,model_params)
A_ij = e_attr(1);
x_j = vattr_j(3);
out_c_ij = A_ij * x_j;

function cbar_i = agg_edge_to_vertex(e_attr)
c_ij = e_attr(:,2);
cbar_i = sum(c_ij);

function out_v_attr = vertex_update(v_attr,cbar_i,g,model_params)
A_ii=v_attr(1); b_i=v_attr(2); x_i=v_attr(3);
omega = g(1);

out_v_attr = x_i + omega*(A_ii \ (b_i - cbar_i));
