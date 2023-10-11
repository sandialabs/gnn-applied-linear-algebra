% test_power_method example

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


function test_power_method(nx)

A=laplacianfun([nx,nx]);
N=size(A,1);
[II,JJ,VV]=find(A);


rng(24601,'twister');
b_exact=rand(N,1);


%% Edge features %%
% 1 immutable: A_ij
e_attr_fixed = VV;
% 1 mutable: 0 input, Aij xj on output
e_attr_mutable = 0*VV;

%% Vertex features
% 1 immutable: x_i
v_attr_fixed = [];
% 1 mutable: b_i, y_i
%   b_i holds the eigenector estimate,
%   y_i holds b_i*b_i
v_attr_mutable = [b_exact, zeros(N, 1)];

%% Global features
% 3 mutable: n, n_A, lambda_max
%   n holds norm(b)
%   n_A holds b^T*A*b
%   lambda_max is the output: the maximal eigenvalue
g_attr_mutable = [0,0,0];

%% Set up feaure cells and run the GNN
% Number of Power Iterations to run
num_iters = 10;

x_fixed={v_attr_fixed,e_attr_fixed,[]};
x_mutable={v_attr_mutable,e_attr_mutable,g_attr_mutable};

for iter = 1:num_iters
    x_mutable = power_iter_gnn(N,II,JJ,x_fixed,x_mutable);
end
output = power_rayleigh_gnn(N,II,JJ,x_fixed,x_mutable);

% Result vector
lambda_nn=output{3}(3);

%% Calculate using traditional formula and compare results
b_trad = b_exact;
for iter = 1:num_iters
    b_trad = A * b_trad;
    b_trad = b_trad/norm(b_trad);
end
lambda_exact = (b_trad' * A * b_trad)/(b_trad' * b_trad);

fprintf('Rel Error = %22.16e\n',norm(lambda_nn - lambda_exact) / norm(lambda_exact));

%% Wrap the iteration layers and rayleigh quotient layers for convenience
function output = power_iter_gnn(N,II,JJ,x_fixed,x_mutable)
% Layer 1
output = gnn(N,II,JJ,x_fixed, x_mutable,...
             @vertex_update_iter_L1, @edge_update_L1, [],...
             1,@agg_edge_to_vertex, [], [], []);
% Layer 2
output = gnn(N,II,JJ,x_fixed, output,...
             @vertex_update_L2, [], @global_update_iter_L2,...
             1,[],[], @agg_vertex_to_global,[]);
% Layer 3
output = gnn(N,II,JJ,x_fixed, output,...
             @vertex_update_iter_L3,[],[],...
             1,[],[],[],[]);

function output = power_rayleigh_gnn(N,II,JJ,x_fixed,x_mutable)
% Layer 1
output = gnn(N,II,JJ,x_fixed,x_mutable,...
             @vertex_update_rq_L1,@edge_update_L1,@global_update_rq_L1,...
             1,@agg_edge_to_vertex,[],@agg_vertex_to_global,[]);

% Layer 2
output = gnn(N,II,JJ,x_fixed,output,...
             @vertex_update_L2,[],@global_update_rq_L2,...
             1,[],[],@agg_vertex_to_global,[]);

%% Aggregation Functions - only two for all layers

%% GNN Operation Ordering:
% c_ij = edge_update(e_attr,vattr_i,vattr_j,g,model_params)
% cbar_i = agg_edge_to_vertex(c_ij)
% v_attr = vertex_update(v_attr,cbar_i,g,model_params)
% e_aggregated = agg_edge_to_global(c_ij)
% v_aggregated = agg_vertex_to_global(v_attr)
% g  = global_update(g,v_aggregated,e_aggregated,model_params)

function cbar_i = agg_edge_to_vertex(e_attr)
c_ij = e_attr(:,2);
cbar_i = sum(c_ij);

function ybar = agg_vertex_to_global(v_attr)
y_i = v_attr(:,2);
ybar = sum(y_i);


% The layer 1 edge update is the same for both iteration layers and rayleigh quotient layers
function out_c_ij = edge_update_L1(e_attr,vattr_i,vattr_j,g,model_params)
A_ij = e_attr(1);
b_j = vattr_j(1);

out_c_ij = A_ij * b_j;

% The layer 2 vertex update is the same for both iteration layers and rayleigh quotient layers
function out_v_attr = vertex_update_L2(v_attr,vbar_i,g,model_params)
b_i = v_attr(1);
y_i = b_i.*b_i;

out_v_attr = [b_i, y_i];

%% Iterative Layers %%
%%%%%%%%%%%%%%%%%%%%%%

%%%% Layer 1
% Layer 1 edge update above

function out_v_attr = vertex_update_iter_L1(v_attr,cbar_i,g,model_params)
y_i = v_attr(2);
b_i = cbar_i;

out_v_attr = [b_i, y_i];

% No global update for Layer 1

%%%% Layer 2
% No edge update for layer 2

% Layer 2 vertex update above

function out_g = global_update_iter_L2(g,v_aggregated,e_aggregated,model_params)
n_A = g(2); lambda_max = g(3);

n = sqrt(v_aggregated);

out_g = [n, n_A, lambda_max];

%%%% Layer 3
% No edge update for layer 3

function out_v_attr = vertex_update_iter_L3(v_attr,cbar_i,g,model_params)
b_i = v_attr(1); y_i = v_attr(2);
n = g(1);
b_i = b_i / n;
out_v_attr = [b_i, y_i];

% No global update for layer 3

%% Rayliegh Quotient Functions %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Layer 1
% Layer 1 edge update above

function out_v_attr = vertex_update_rq_L1(v_attr,cbar_i,g,model_params)
b_i = v_attr(1);
y_i = b_i.*cbar_i;

out_v_attr = [b_i, y_i];

function out_g = global_update_rq_L1(g,v_aggreated,e_aggregated,model_params)
n = g(1); lambda_max = g(3);

n_A = v_aggreated;

out_g = [n, n_A, lambda_max];

%%%% Layer 2
% No edge update for layer 2

% Layer 2 vertex update above

function out_g = global_update_rq_L2(g, v_aggregated, e_aggregated, model_params)
n = g(1); n_A = g(2);

lambda_max = n_A / v_aggregated;

out_g = [n, n_A, lambda_max];
