% function [K,M]=heateqnfem2dfun(num_cells,h,[bcs])
% 2D Finite element matrix with BCs 
% - num_cells: 2-vector of # cells in each dimension 
% - h:         2-vector of grid spacing (h) in each dimension
% - bcs:       2-vector of bcs for each dim (0=Neuman,
%              1=Ones-and-Zeros Dirichlet, 2=Eliminated Dirichlet).
%              Defaults to onez-and-zeros Dirichlet everywhere
% Output:
% - K:         Stiffness matrix
% - M:         Currently empty

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


function [K,M]=heateqnfem2dfun(num_cells,h_all,bcs)
if(~exist('bcs','var')), bcs=[1,1]; end

% Sanity checks
if(length(num_cells)~=2 || length(h_all)~=2 || length(bcs)~=2),
  fprintf('heateqnfem2dfun: need length-2 vectors for num_cells, h and bcs\n');
  return
end
if( (bcs(1)==2 && bcs(2)~=2) || (bcs(1)~=2 && bcs(2)==2) ),
  fprintf('heateqnfem2dfun: eliminated Dirchlet needs to be set for all BCs\n');
  return
end


% Stretch factor
h=h_all(1);
alpha=h_all(2) / h;

% Element functions
xneighbor = [0,1,0,0;...
             1,0,0,0;...
             0,0,0,1;...
             0,0,1,0];
yneighbor = [0,0,0,1;...
             0,0,1,0;...
             0,1,0,0;...
             1,0,0,0];
cneighbor = [0,0,1,0;...
             0,0,0,1;...
             1,0,0,0;...
             0,1,0,0];

% Stencil values are taken correspond to a 2D quad discretization
% and are taken from "A cut-based distance Laplacian dropping criterion 
% for smoothed aggregation multigrid" by. C. Siefert, D. Sunderland
% and R. Tuminaro, 2022.
%
%
% Self, x, y cell
element_vals =(1/6/alpha)* [2*alpha^2+2, -2*alpha^2+1, alpha^2-2, -1-alpha^2];

% Local cell ordering
% 4 o----o 3
%   |    |
%   |    |
% 1 o----o 2

ELEMENT_K = element_vals(1)*eye(4,4) + element_vals(2)*xneighbor + ...
    element_vals(3)*yneighbor + element_vals(4)*cneighbor;


% Interior stencil for checking
% Self, far, cell, close
Astencil = @(alpha,h)(1./(6*alpha).*[4*(2*alpha.^2+2),...
                    2*(alpha.^2-2),...
                    (-alpha.^2-1),...
                    2*(-2*alpha.^2+1)]);
Kcompare = Astencil(alpha,h);
Kcompare = Kcompare([1,4,3,2]);


% FEM assembly
Npts = prod(1+num_cells);
K=spalloc(Npts,Npts,Npts*9);

for YID=1:num_cells(2),  
  for XID=1:num_cells(1),
    base = (num_cells(1)+1)*(YID-1) + XID;    
    GIDX=[base,base+1,base+num_cells(1)+2,base+num_cells(1)+1];
    %fprintf('cell (%d,%d) indices = [%d,%d,%d,%d]\n',XID,YID,GIDX(1),GIDX(2),GIDX(3),GIDX(4));
    K(GIDX,GIDX) = K(GIDX,GIDX) + ELEMENT_K;
       
  end
end


% Sanity checking
%base=num_cells(1)+3;
%GIDX=[base,base+1,base+num_cells(1)+2,base+num_cells(1)+1];
%fprintf('Actual  = ');fprintf('%6.4e ',full(K(base,GIDX)));fprintf('\n');
%fprintf('Compare = ');fprintf('%6.4e ',Kcompare);fprintf('\n');


% Boundary conditions
BOTTOM_IDX=1:num_cells(1)+1;
TOP_IDX=(num_cells(1)+1)*num_cells(2)+1:Npts;
LEFT_IDX=1:num_cells(1)+1:(num_cells(1)+1)*num_cells(2)+1;
RIGHT_IDX=num_cells(1)+1:num_cells(1)+1:Npts;
Ntb = length(TOP_IDX);
Nlr = length(LEFT_IDX);

% OAZ Dirichlet
if(bcs(1) == 1),
  K(LEFT_IDX,:)  = sparse(Nlr,Npts);
  K(RIGHT_IDX,:) = sparse(Nlr,Npts);
  K(:,LEFT_IDX)  = sparse(Npts,Nlr);
  K(:,RIGHT_IDX) = sparse(Npts,Nlr);

  K(LEFT_IDX,LEFT_IDX)   = speye(Nlr,Nlr);
  K(RIGHT_IDX,RIGHT_IDX) = speye(Nlr,Nlr);
end
if(bcs(2) == 1),
  K(TOP_IDX,:)    = sparse(Ntb,Npts);
  K(BOTTOM_IDX,:) = sparse(Ntb,Npts);
  K(:,TOP_IDX)    = sparse(Npts,Ntb);
  K(:,BOTTOM_IDX) = sparse(Npts,Ntb);
  
  K(TOP_IDX,TOP_IDX)       = speye(Ntb,Ntb);
  K(BOTTOM_IDX,BOTTOM_IDX) = speye(Ntb,Ntb);
end

% Eliminated Dirichlet
if(bcs(1) == 2),
  BC_IDX=unique(sort([LEFT_IDX,RIGHT_IDX,TOP_IDX,BOTTOM_IDX]));
  INTERIOR_IDX=setdiff(1:Npts,BC_IDX);
  K = K(INTERIOR_IDX,INTERIOR_IDX);  
end

  

M=[];