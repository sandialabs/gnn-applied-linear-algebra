% test_direct_interpolation

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

function test_direct_interpolation(nx)
%  Direct interpolation:
%   direct interpolation
%
omega = 1;

A=laplacianfun([nx,nx]);
N=size(A,1);
D=spdiags(diag(A),0,N,N);

% remove diagonal - makes the max_{i\neq j} easier
%  if the diagonals are not included in the edge features
L = A - D;

[II,JJ,VV]=find(L);

% Strength of connection
S = spones(A);
S_no_diag = S - diag(diag(S));
[SI,SJ,SV]=find(S_no_diag); % this has to match sparsity pattern of A

% Coarse/fine splitting
C = zeros(N,1);
C(1:2:end) = 1;

%% Edge features %%
% 2 immutable: A_ij, S_ij
e_attr_fixed = [VV,SV];
% 1 mutable: 0 input, w_ij on output
e_attr_mutable = 0*VV;


%% Vertex features
% 2 immutable: A_ii, C_i
v_attr_fixed = [diag(A), C];
% 1 mutable: 0 input, alpha_i on output
v_attr_mutable = 0*diag(A);

x_fixed={v_attr_fixed,e_attr_fixed,[]};
x_mutable={v_attr_mutable,e_attr_mutable,[]};

output = gnn(N,II,JJ,x_fixed,x_mutable,...
             @vertex_update_L1,@edge_update_L1,[],...
             1,@agg_edge_to_vertex_L1,[],[],[]);

output = gnn(N,II,JJ,x_fixed,output,...
             [],@edge_update_L2,[],...
             1,[],[],[],[]);

vals = output{2}; % get values of w from output

P = sparse(II,JJ,vals);
P = P + diag(ones(N,1)); % place ones on diagonal;
P = P(:,find(C)); % remove fine cols

disp(size(P));
disp(full(P));


%% GNN Operation Ordering:
% w_ij = edge_update(e_attr,vattr_i,vattr_j,g,model_params)
% gammabar_i = agg_edge_to_vertex(z_ij)
% alpha_i = vertex_update(v_attr,zbar_i,g,model_params)
% e_aggregated = agg_edge_to_global(z_ij)
% v_aggregated = agg_vertex_to_global(v_attr)
% g  = global_update(g,v_aggregated,e_aggregated,model_params)


% layer 1
function out_w_ij = edge_update_L1(e_attr,vattr_i,vattr_j,g,model_params)
C_j = vattr_j(2);
out_w_ij = C_j;

function gammabar_i = agg_edge_to_vertex_L1(e_attr)
A_ik = e_attr(:,1); S_ik = e_attr(:,2); v_ik = e_attr(:,3);
gammabar_i = sum(A_ik)/sum(A_ik.*v_ik.*S_ik);

function out_v_attr = vertex_update_L1(v_attr,gammabar_i,g,model_params)
A_ii = v_attr(1);
out_v_attr = gammabar_i/A_ii;

% layer 2
function out_w_ij=edge_update_L2(e_attr,vattr_i,vattr_j,g,model_params)
C_i = vattr_i(1,2);
A_ij = e_attr(1,1);
alpha_i = vattr_i(1,3);

out_w_ij = (1-C_i)*(-A_ij*alpha_i);
if C_i == 1 % Extra care to handle INF and NAN in alpha_i
    out_w_ij = 0;
end
