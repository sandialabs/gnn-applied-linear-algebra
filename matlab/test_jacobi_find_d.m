% test_jacobi_find_d example

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


function test_jacobi_find_d

%%%%%%%%%%%%% Initial single-matrix testing %%%%%%%%%%%%% 

nx=10;
A=laplacianfun([nx,nx]);
N=size(A,1);
D=spdiags(diag(A),0,N,N);
omega=2/3; % for giggles
[II,JJ,VV]=find(A);


rng(24601,'twister');


model_parameters = struct;

%% Edge features %%
% 1 immutable: A_ij
e_attr_fixed = VV;
% 0 mutable
e_attr_mutable = [];


%% Vertex features
% 1 immutable: A_i
v_attr_fixed = [diag(A)];
% 1 mutable: 0 on input, diagonal value on output
v_attr_mutable = zeros(N,1);


% Vertex model parameters;
% 4 agg edge features: e_min, e_mean, e_sum, e_max
v_agg_edge_features = 4;

output_size_per_level=[3,3,1];
v_weights = {};

in_size=size(v_attr_fixed,2) + size(v_attr_mutable,2) + v_agg_edge_features;
for I=1:length(output_size_per_level),
  out_size = output_size_per_level(I);
  v_weights{I} = {dlarray(rand(out_size,in_size)), dlarray(rand(out_size,1))};
  in_size = out_size;
end
model_parameters.vertex = v_weights;

%% Global features
% 1 immutable: omega
g_attr_fixed = omega;
% 0 mutable
g_attr_mutable = [];


%% Define the GNN model
x_fixed={v_attr_fixed,e_attr_fixed,g_attr_fixed};
x_mutable={v_attr_mutable,e_attr_mutable,g_attr_mutable};

output = gnn(N,II,JJ,x_fixed,x_mutable,...
             @vertex_update,[],[],...
             v_agg_edge_features,@agg_edge_to_vertex,[],[],model_parameters);



%% Evaluate the loss function
df=damping_factor(A,omega,output{1});

fprintf('Using NN diagonal, damping factor = %6.4e\n',df);

% Result vector
%x1_nn=output{1};

% Jacobi
%x_k+1 = x_k + w D^{-1} ( b-A x_k)
%x1_exact = x0_exact + omega * (D \ (b_exact - A * x0_exact));


%fprintf('Rel Error = %22.16e\n',norm(x1_nn - x1_exact) / norm(x1_exact));




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% e_ij = update_edge(e_ij,v_i,v_j,g,model_params)
% ehat_i = agg_edge_to_vertex(e_i*)
% n_i = vertex_update(n_i,ehat_i,g,model_params)
% eg = agg_edge_to_global(edges)
% vg = agg_vertex_to_global(edges)
% g  = update_global(g,vg,eg,model_params)
function ehat_i = agg_edge_to_vertex(e_istar)
ehat_i = [min(e_istar(:,1)),mean(e_istar(:,1)),sum(e_istar(:,1)),max(e_istar(:,1))];


% 1 immutable: A_ii
% 1 mutable: D_i on output
% NN params
function out_n_i= vertex_update(n_i,ehat_i,g,model_parameters)
% Just use a sequential NN to update the mutable variable
out_n_i = sequential_nn([n_i,ehat_i],model_parameters.vertex);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function df=damping_factor(A,omega,diagonal_value);
N=size(A,1);
D=spdiags(diagonal_value,0,N,N);

J = eye(N,N) - omega * (D\A);


opts=struct; opts.tol=1e-6;
df = eigs(A,1,'largestabs',opts);

