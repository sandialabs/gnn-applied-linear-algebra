% test_classic_soc example

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


function test_classic_soc(nx)
%  Classic-style SOC:
%   S_ij = A_ij / max(-A_ik, k \neq i)
%   strongly connected if S_ij \geq \theta
%

A=laplacianfun([nx,nx]);
N=size(A,1);

D=spdiags(diag(A),0,N,N);

% remove diagonal - makes the max_{i\neq j} easier
%  if the diagonals are not included in the edge features
L = A - D;

[II,JJ,VV]=find(L);

disp(size(II))
disp(size(JJ))

%% Edge features %%
% 1 immutable: A_ij
e_attr_fixed = VV;
% 1 mutable: 0 input, S_ij on output
e_attr_mutable = 0*VV;

%% Vertex features
% 0 immutable
v_attr_fixed = [];
% 1 mutable, 0 input, v_i on output
v_attr_mutable = zeros(N,1);

x_fixed={v_attr_fixed,e_attr_fixed,[]};
x_mutable={v_attr_mutable,e_attr_mutable,[]};

global theta;
theta = 0.25;

disp(x_fixed)
disp(x_mutable)

output = gnn(N,II,JJ,x_fixed,x_mutable,...
             @vertex_update,[],[],...
             1,@agg_edge_to_vertex,[],[],[]);

output = gnn(N,II,JJ,x_fixed,output,...
             [],@edge_update,[],...
             1,[],[],[],[]);

% Result vector
VV_nn=output{2};
SOC_nn = sparse(II,JJ,VV_nn);


% Analytic
M = max(-A,[],2);
SOC_exact = relu(bsxfun(@rdivide,-L,M(:)) - theta);
% SOC_exact = bsxfun(@rdivide,-L,M(:));


SOC_nn_D =spdiags(diag(SOC_nn),0,N,N);
SOC_nn = SOC_nn - SOC_nn_D; % remove diagonal


fprintf('Rel Error = %22.16e\n',norm(SOC_nn - SOC_exact,inf) / norm(SOC_exact,inf));

%% GNN Operation Ordering:
% z_ij = edge_update(e_attr,vattr_i,vattr_j,g,model_params)
% zbar_i = agg_edge_to_vertex(z_ij)
% v_attr = vertex_update(v_attr,zbar_i,g,model_params)
% e_aggregated = agg_edge_to_global(z_ij)
% v_aggregated = agg_vertex_to_global(v_attr)
% g  = global_update(g,v_aggregated,e_aggregated,model_params)

function out_v_attr = vertex_update(v_attr,cbar_i,g,model_params)
out_v_attr = cbar_i;

function out_S_ij = edge_update(e_attr,vattr_i,vattr_j,g,model_params)
global theta;
A_ij = e_attr(1);
v_i = vattr_i(1);
out_S_ij = relu(-A_ij / v_i - theta);
% out_S_ij = -A_ij / v_i;


function cbar_i = agg_edge_to_vertex(e_attr)
A_ij = e_attr(1);
cbar_i = max(-A_ij);

function out = relu(A)
out = A;
for ii = 1:size(A,1)
    for jj = 1:size(A,2)
        if out(ii,jj) < 0
            out(ii,jj) = 0;
        end
    end
end
