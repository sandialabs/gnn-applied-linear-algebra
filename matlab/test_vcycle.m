% test_vcycle example
%  V Cycle:
%   Perform a V-cycle using the GNN components
%   constructed for basic linear algebra operations


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


function test_vcycle(nx)
%  V Cycle:
%   Perform a V-cycle using the GNN components
%   constructed for basic linear algebra operations

omega=2/3;


A=laplacianfun([nx,nx]);
N=size(A,1);
[II,JJ,VV]=find(A);
A_no_diag = A - diag(diag(A));
[II_no_diag,JJ_no_diag,VV_no_diag]=find(A_no_diag);

rng(24601,'twister');
x0_exact=rand(N,1);
b_exact=rand(N,1);

% Coarse/Fine Splitting
C = zeros(N,1);
C(1:2:end) = 1;

% SOC
e_attr_fixed = VV_no_diag;
e_attr_mutable = 0*VV_no_diag;
v_attr_fixed = [];
v_attr_mutable = zeros(N,1);

[VV_nn, SOC_nn] = gnn_soc(N,II_no_diag,JJ_no_diag,v_attr_fixed,v_attr_mutable,e_attr_fixed,e_attr_mutable);


%% Direct Interpolation
%TODO: use the nn created SOC. Have to worry about sparsity pattern mismatch
%[SI,SJ,SV]=find(SOC_nn(find(A_no_diag))); % this has to match sparsity pattern of A
[SI,SJ,SV]=find(spones(A_no_diag));
e_attr_fixed = [VV_no_diag,SV];
e_attr_mutable = 0*VV_no_diag;
v_attr_fixed = [diag(A), C];
v_attr_mutable = 0*diag(A);

P = gnn_direct_interp(C,N,II_no_diag,JJ_no_diag,v_attr_fixed,v_attr_mutable,e_attr_fixed,e_attr_mutable);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("V-cycle residual");
x0_in = x0_exact;
disp(norm(b_exact-A*x0_exact));
x1_nn = gnn_vcycle(N,II,JJ,VV,A,b_exact,x0_in,P,omega);
disp(norm(b_exact-A*x1_nn));
for i = 1:2,
  x1_nn = gnn_vcycle(N,II,JJ,VV,A,b_exact,x1_nn,P,omega);
  disp(norm(b_exact-A*x1_nn));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("Just Jacobi residual");
disp(norm(b_exact-A*x0_exact));
%% Jacobi
e_attr_fixed = VV;
e_attr_mutable = 0*VV;
v_attr_fixed = [diag(A), b_exact];
v_attr_mutable = x0_exact;
g_attr_fixed = omega;
x_fixed={v_attr_fixed,e_attr_fixed,g_attr_fixed};
x_mutable={v_attr_mutable,e_attr_mutable,[]};

output = gnn(N,II,JJ,x_fixed,x_mutable,...
             @jacobi_vertex_update,@jacobi_edge_update,[],...
             1,@jacobi_agg_edge_to_vertex,[],[],[]);

% Result vector
x1_nn=output{1};


R1 = norm(b_exact-A*x1_nn);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Jacobi
e_attr_fixed = VV;
e_attr_mutable = 0*VV;
v_attr_fixed = [diag(A), b_exact];
v_attr_mutable = x1_nn;%%%%%%%%
g_attr_fixed = omega;
x_fixed={v_attr_fixed,e_attr_fixed,g_attr_fixed};
x_mutable={v_attr_mutable,e_attr_mutable,[]};

output = gnn(N,II,JJ,x_fixed,x_mutable,...
             @jacobi_vertex_update,@jacobi_edge_update,[],...
             1,@jacobi_agg_edge_to_vertex,[],[],[]);

% Result vector
x1_nn=output{1};


disp([R1, norm(b_exact-A*x1_nn)]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Jacobi
e_attr_fixed = VV;
e_attr_mutable = 0*VV;
v_attr_fixed = [diag(A), b_exact];
v_attr_mutable = x1_nn;%%%%%%%%
g_attr_fixed = omega;
x_fixed={v_attr_fixed,e_attr_fixed,g_attr_fixed};
x_mutable={v_attr_mutable,e_attr_mutable,[]};

output = gnn(N,II,JJ,x_fixed,x_mutable,...
             @jacobi_vertex_update,@jacobi_edge_update,[],...
             1,@jacobi_agg_edge_to_vertex,[],[],[]);

% Result vector
x1_nn=output{1};


R1 = (norm(b_exact-A*x1_nn));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Jacobi
e_attr_fixed = VV;
e_attr_mutable = 0*VV;
v_attr_fixed = [diag(A), b_exact];
v_attr_mutable = x1_nn;%%%%%%%%
g_attr_fixed = omega;
x_fixed={v_attr_fixed,e_attr_fixed,g_attr_fixed};
x_mutable={v_attr_mutable,e_attr_mutable,[]};

output = gnn(N,II,JJ,x_fixed,x_mutable,...
             @jacobi_vertex_update,@jacobi_edge_update,[],...
             1,@jacobi_agg_edge_to_vertex,[],[],[]);

% Result vector
x1_nn=output{1};


disp([R1, norm(b_exact-A*x1_nn)]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Jacobi
e_attr_fixed = VV;
e_attr_mutable = 0*VV;
v_attr_fixed = [diag(A), b_exact];
v_attr_mutable = x1_nn;%%%%%%%%
g_attr_fixed = omega;
x_fixed={v_attr_fixed,e_attr_fixed,g_attr_fixed};
x_mutable={v_attr_mutable,e_attr_mutable,[]};

output = gnn(N,II,JJ,x_fixed,x_mutable,...
             @jacobi_vertex_update,@jacobi_edge_update,[],...
             1,@jacobi_agg_edge_to_vertex,[],[],[]);

% Result vector
x1_nn=output{1};


R1 = (norm(b_exact-A*x1_nn));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Jacobi
e_attr_fixed = VV;
e_attr_mutable = 0*VV;
v_attr_fixed = [diag(A), b_exact];
v_attr_mutable = x1_nn;%%%%%%%%
g_attr_fixed = omega;
x_fixed={v_attr_fixed,e_attr_fixed,g_attr_fixed};
x_mutable={v_attr_mutable,e_attr_mutable,[]};

output = gnn(N,II,JJ,x_fixed,x_mutable,...
             @jacobi_vertex_update,@jacobi_edge_update,[],...
             1,@jacobi_agg_edge_to_vertex,[],[],[]);

% Result vector
x1_nn=output{1};


disp([R1, norm(b_exact-A*x1_nn)]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% GNN Operations:

%% Jacobi
function out_c_ij = jacobi_edge_update(e_attr,vattr_i,vattr_j,g,model_params)
  A_ij = e_attr(1);
  x_j = vattr_j(3);
  out_c_ij = A_ij * x_j;

function cbar_i = jacobi_agg_edge_to_vertex(e_attr)
  c_ij = e_attr(:,2);
  cbar_i = sum(c_ij);

function out_v_attr = jacobi_vertex_update(v_attr,cbar_i,g,model_params)
  D_i=v_attr(1); b_i=v_attr(2); x_i=v_attr(3);
  omega = g(1);
  out_v_attr = x_i + omega*(D_i \ (b_i - cbar_i));


%% matvec
function out_c_ij = matvec_edge_update(e_attr,vattr_i,vattr_j,g,model_params)
  A_ij = e_attr(1);
  x_j = vattr_j(1);
  out_c_ij = A_ij * x_j;

function cbar_i = matvec_agg_edge_to_vertex(e_attr)
  c_ij = e_attr(:,2);
  cbar_i = sum(c_ij);

function out_v_attr = matvec_vertex_update(v_attr,cbar_i,g,model_params)
  out_v_attr = cbar_i;


%% SOC
function out_v_attr = soc_vertex_update(v_attr,cbar_i,g,model_params)
  out_v_attr = cbar_i;

function out_S_ij = soc_edge_update(e_attr,vattr_i,vattr_j,g,model_params)
  A_ij = e_attr(1);
  v_i = vattr_i(1);
  out_S_ij = -A_ij / v_i;

function cbar_i = soc_agg_edge_to_vertex(e_attr)
  A_ij = e_attr(1);
  cbar_i = max(-A_ij);

function [VV_nn, SOC_nn] = gnn_soc(N,II_no_diag,JJ_no_diag,v_attr_fixed,v_attr_mutable,e_attr_fixed,e_attr_mutable)
  x_fixed={v_attr_fixed,e_attr_fixed,[]};
  x_mutable={v_attr_mutable,e_attr_mutable,[]};

  output = gnn(N,II_no_diag,JJ_no_diag,x_fixed,x_mutable,...
                @soc_vertex_update,[],[],...
                1,@soc_agg_edge_to_vertex,[],[],[]);

  output = gnn(N,II_no_diag,JJ_no_diag,x_fixed,output,...
                [],@soc_edge_update,[],...
                1,[],[],[],[]);

  % Result vector
  VV_nn=output{2};
  SOC_nn = sparse(II_no_diag,JJ_no_diag,VV_nn);


%% Direct Interpolation
% layer 1
function out_w_ij = interp_edge_update_L1(e_attr,vattr_i,vattr_j,g,model_params)
  C_j = vattr_j(2);
  out_w_ij = C_j;

function gammabar_i = interp_agg_edge_to_vertex_L1(e_attr)
  A_ik = e_attr(:,1); S_ik = e_attr(:,2); w_ik = e_attr(:,3);
  gammabar_i = sum(A_ik)/sum(A_ik.*w_ik.*S_ik);

function out_v_attr = interp_vertex_update_L1(v_attr,gammabar_i,g,model_params)
  D_i = v_attr(1);
  out_v_attr = gammabar_i/D_i;

% layer 2
function out_w_ij=interp_edge_update_L2(e_attr,vattr_i,vattr_j,g,model_params)
  C_i = vattr_i(1,2);
  A_ij = e_attr(1,1);
  alpha_i = vattr_i(1,3);

  out_w_ij = (1-C_i)*(-A_ij*alpha_i);
  if C_i == 1 % Extra care to handle INF and NAN in alpha_i
    out_w_ij = 0;
  end

% layer 1
function P = gnn_direct_interp(C_split,N,II_no_diag,JJ_no_diag,v_attr_fixed,v_attr_mutable,e_attr_fixed,e_attr_mutable)
    x_fixed={v_attr_fixed,e_attr_fixed,[]};
    x_mutable={v_attr_mutable,e_attr_mutable,[]};

    output = gnn(N,II_no_diag,JJ_no_diag,x_fixed,x_mutable,...
                 @interp_vertex_update_L1,@interp_edge_update_L1,[],...
                 1,@interp_agg_edge_to_vertex_L1,[],[],[]);

    output = gnn(N,II_no_diag,JJ_no_diag,x_fixed,output,...
                 [],@interp_edge_update_L2,[],...
                 1,[],[],[],[]);

    vals = output{2}; % get values of P from output

    P = sparse(II_no_diag,JJ_no_diag,vals);
    P = P + diag(ones(N,1)); % place ones on diagonal;
    P = P(:,find(C_split)); % remove fine cols using coarse/fine splitting


%% V-cycle
function x1_nn = gnn_vcycle(N,II,JJ,VV,A,b_exact,x0_in,P,omega)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Jacobi
    e_attr_fixed = VV;
    e_attr_mutable = 0*VV;
    v_attr_fixed = [diag(A), b_exact];
    v_attr_mutable = x0_in;
    g_attr_fixed = omega;
    x_fixed={v_attr_fixed,e_attr_fixed,g_attr_fixed};
    x_mutable={v_attr_mutable,e_attr_mutable,[]};

    output = gnn(N,II,JJ,x_fixed,x_mutable,...
                 @jacobi_vertex_update,@jacobi_edge_update,[],...
                 1,@jacobi_agg_edge_to_vertex,[],[],[]);

    % Result vector
    x1_nn=output{1};

    %% Residual
    e_attr_fixed = VV;
    e_attr_mutable = 0*VV;
    v_attr_fixed = x1_nn;
    v_attr_mutable = 0*x1_nn;
    x_fixed={v_attr_fixed,e_attr_fixed,[]};
    x_mutable={v_attr_mutable,e_attr_mutable,[]};

    output = gnn(N,II,JJ,x_fixed,x_mutable,...
                 @matvec_vertex_update,@matvec_edge_update,[],...
                 1,@matvec_agg_edge_to_vertex,[],[],[]);

    % Result vector
    r_nn=b_exact - output{1};

    %% Coarse Grid
    Ac = P'*A*P;

    rc = P'*r_nn;

    %% Coarse Grid Solve
    ec = Ac\rc;

    %% Project Coarse Grid Correction
    coarse_grid_correction = P*ec;

    x_new = x1_nn + coarse_grid_correction;

    %% Jacobi
    e_attr_fixed = VV;
    e_attr_mutable = 0*VV;
    v_attr_fixed = [diag(A), b_exact];
    v_attr_mutable = x_new;%%%%%%%%
    g_attr_fixed = omega;
    x_fixed={v_attr_fixed,e_attr_fixed,g_attr_fixed};
    x_mutable={v_attr_mutable,e_attr_mutable,[]};

    output = gnn(N,II,JJ,x_fixed,x_mutable,...
                 @jacobi_vertex_update,@jacobi_edge_update,[],...
                 1,@jacobi_agg_edge_to_vertex,[],[],[]);

    % Result vector
    x1_nn=output{1};
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
