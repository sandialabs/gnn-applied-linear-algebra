% test_matrix_weighted_norm example

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


function test_matrix_weighted_norm(nx)

A=laplacianfun([nx,nx]);
N=size(A,1);
[II,JJ,VV]=find(A);


rng(24601,'twister');
x_exact=rand(N,1);


%% Edge features %%
% 1 immutable: A_ij
e_attr_fixed = VV;
% 1 mutable: 0 input, Aij xj on output
e_attr_mutable = 0*VV;

%% Vertex features
% 1 immutable: x_i
v_attr_fixed = x_exact;
% 1 mutable: y_i holds x_i A_ij x_j
v_attr_mutable = zeros(size(x_exact));

%% Global features
% 1 mutable: for output, contains sqrt(x^T A x) on output
g_attr_mutable = 0;

%% Set up feaure cells and run the GNN

x_fixed={v_attr_fixed,e_attr_fixed,[]};
x_mutable={v_attr_mutable,e_attr_mutable,g_attr_mutable};

output = gnn(N,II,JJ,x_fixed,x_mutable,...
             @vertex_update,@edge_update,@global_update,...
             1,@agg_edge_to_vertex,[],@agg_vertex_to_global,[]);

% Result vector
b_nn=output{3};

%% Calculate using the formula and compare results
b_exact = sqrt(x_exact'*A*x_exact);

fprintf('Rel Error = %22.16e\n',norm(b_nn - b_exact) / norm(b_exact));

%% GNN Operation Ordering:
% c_ij = edge_update(e_attr,vattr_i,vattr_j,g,model_params)
% cbar_i = agg_edge_to_vertex(c_ij)
% v_attr = vertex_update(v_attr,zbar_i,g,model_params)
% e_aggregated = agg_edge_to_global(c_ij)
% v_aggregated = agg_vertex_to_global(v_attr)
% g  = global_update(g,v_aggregated,e_aggregated,model_params)

function out_c_ij=edge_update(e_attr,vattr_i,vattr_j,g,model_params)
W_ij = e_attr(1);
x_j = vattr_j(1);
out_c_ij = W_ij * x_j;

function cbar_i = agg_edge_to_vertex(e_attr)
c_ij = e_attr(:,2);
cbar_i = sum(c_ij);

function ybar = agg_vertex_to_global(v_attr)
y_i = v_attr(:,2);
ybar = sum(y_i);

function out_v_attr = vertex_update(v_attr,cbar_i,g,model_params)
x_i = v_attr(1);
out_v_attr = x_i*cbar_i;

function out_g = global_update(g, v_aggregated, e_aggregated, model_params)
ybar = v_aggregated;
out_g = sqrt(ybar);
