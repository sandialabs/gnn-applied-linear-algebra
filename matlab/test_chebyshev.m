% test_chebysev example

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


function test_chebyshev(nx)

A=laplacianfun([nx,nx]);
N=size(A,1);
D=spdiags(diag(A),0,N,N);

[II,JJ,VV]=find(A);

eig_min = eigs(A,1,'smallestabs');
eig_max = eigs(A,1,'largestabs');
d = (eig_max+eig_min)/2;
c = (eig_max-eig_min)/2;



rng(24601,'twister');
x0_exact=rand(N,1);
b_exact=rand(N,1);
%x0_exact=A\b_exact;


%% Edge features %%
% 1 immutable: A_ij
e_attr_fixed = VV;
% 1 mutable: 0 input, Aij xj on output
e_attr_mutable = 0*VV;

%% Vertex features
% 1 immutable: b_i
v_attr_fixed = [b_exact];
% 3 mutable: input x_i and placeholders for r_i, p_i
v_attr_mutable = [x0_exact, -ones(size(x0_exact)), -ones(size(x0_exact))];

%% Global features
% 2 immutable: c, d
g_attr_fixed = [c, d];
% 2 mutable: alpha, beta - placeholders, will be assigned in the GNN
g_attr_mutable = [0,0];


x_fixed={v_attr_fixed,e_attr_fixed,g_attr_fixed};
x_mutable={v_attr_mutable,e_attr_mutable,g_attr_mutable};

disp(norm(A*x0_exact-b_exact))
degree = 5;
out = cheby_gnn(degree,N,II,JJ,x_fixed,x_mutable,b_exact,A);
out = cheby_gnn(degree,N,II,JJ,x_fixed,out,b_exact,A);
out = cheby_gnn(degree,N,II,JJ,x_fixed,out,b_exact,A);
out = cheby_gnn(degree,N,II,JJ,x_fixed,out,b_exact,A);
out = cheby_gnn(degree,N,II,JJ,x_fixed,out,b_exact,A);
out = cheby_gnn(degree,N,II,JJ,x_fixed,out,b_exact,A);
out = cheby_gnn(degree,N,II,JJ,x_fixed,out,b_exact,A);


%% Wrap the GNN layers for chebyshev nicely in one function
function out = cheby_gnn(degree, N,II,JJ,x_fixed,x_mutable,b_exact,A)
output = gnn(N,II,JJ,x_fixed,x_mutable,...
             @vertex_update_I1_L1,@edge_update_I1_L1,@global_update_I1_L1,...
             1,@agg_edge_to_vertex_L1,[],[],[]);

output = gnn(N,II,JJ,x_fixed,output,...
             @vertex_update_L2,[],[],...
             1,[],[],[],[]);
if degree > 1
    output = gnn(N,II,JJ,x_fixed,output,...
                 @vertex_update_L1,@edge_update_L1,@global_update_I2_L1,...
                 1,@agg_edge_to_vertex_L1,[],[],[]);
    output = gnn(N,II,JJ,x_fixed,output,...
                 @vertex_update_L2,[],[],...
                 1,[],[],[],[]);
    for deg = 3:degree
        output = gnn(N,II,JJ,x_fixed,output,...
                     @vertex_update_L1,@edge_update_L1,@global_update_L1,...
                     1,@agg_edge_to_vertex_L1,[],[],[]);
        output = gnn(N,II,JJ,x_fixed,output,...
                     @vertex_update_L2,[],[],...
                     1,[],[],[],[]);
    end
end
disp(norm(A*output{1}(:,1)-b_exact))

out = output;

%% GNN Operation Ordering:
% z_ij = edge_update(e_attr,vattr_i,vattr_j,g,model_params)
% zbar_i = agg_edge_to_vertex(z_ij)
% v_attr = vertex_update(v_attr,zbar_i,g,model_params)
% e_aggregated = agg_edge_to_global(z_ij)
% v_aggregated = agg_vertex_to_global(v_attr)
% g  = global_update(g,v_aggregated,e_aggregated,model_params)


%%% The same edge to vertex aggregation is used in layer 1 of all iterations
function zbar_i = agg_edge_to_vertex_L1(e_attr)
  z_ij = e_attr(:,2);
  zbar_i = sum(z_ij);

%%% Edge Updates %%%%
%%%%%%%%%%%%%%%%%%%%%
%%% Iteration 1 Layer 1
function out_z_ij = edge_update_I1_L1(e_attr,vattr_i,vattr_j,g,model_params)
  A_ij = e_attr(1);
  x_j = vattr_j(2);

  out_z_ij = A_ij * x_j;

%%% Iteration > 1, Layer 1
function out_z_ij = edge_update_L1(e_attr,vattr_i,vattr_j,g,model_params)
  A_ij = e_attr(1);
  p_j = vattr_j(4);

  out_z_ij = A_ij * p_j;

%%% Vertex Updates %%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%% Iteration 1, Layer 1
function out_v_attr = vertex_update_I1_L1(v_attr,zbar_i,g,model_params)
  b_i=v_attr(1); x_i=v_attr(2); p_i=v_attr(4);

  r_i = b_i - zbar_i;

  out_v_attr = [x_i, r_i, p_i];

%%% Iteration > 1, Layer 1
function out_v_attr = vertex_update_L1(v_attr,zbar_i,g,model_params)
  b_i = v_attr(1); x_i = v_attr(2); r_i = v_attr(3); p_i = v_attr(4);
  alpha = g(3);

  r_i = r_i - alpha * zbar_i;

  out_v_attr = [x_i, r_i, p_i];

%%% Assuming beta is initialized to 0
%%% All Iterations, Layer 2
function out_v_attr = vertex_update_L2(v_attr,zbar_i,g,model_params)
  b_i = v_attr(1); x_i = v_attr(2); r_i = v_attr(3); p_i = v_attr(4);
  alpha = g(3); beta = g(4);

  p_i = r_i + beta*p_i;
  x_i = x_i + alpha*p_i;

  out_v_attr = [x_i, r_i, p_i];

%%% Global Updates %%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%% Iteration 1, Layer 1
function out_g = global_update_I1_L1(g,v_aggreated,e_eggregated,model_params)
  c = g(1); d=g(2); alpha = g(3); beta = g(4);

  alpha = 1/d;

  out_g = [alpha, beta];

%%% Iteration 2, Layer 1
function out_g  = global_update_I2_L1(g,v_aggregated,e_aggregated,model_params)
  c = g(1); d=g(2); alpha = g(3); beta = g(4);

  beta = 0.5*(c*alpha)^2;
  alpha = 1/(d-beta/alpha);

  out_g = [alpha, beta];

%%% Iteration > 2, Layer 1
function out_g  = global_update_L1(g,v_aggregated,e_aggregated,model_params)
c = g(1); d=g(2); alpha = g(3); beta = g(4);

beta = (c*alpha/2)^2;
alpha = 1/(d-beta/alpha);

out_g = [alpha, beta];
