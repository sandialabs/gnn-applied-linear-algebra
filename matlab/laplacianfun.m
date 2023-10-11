% function [MAT,VERTICES,DN]=laplacianfun(NPTS,[BCS])
%
% Generates a discretized Laplacian operator in any arbitrarily
% large number of dimensions.
% Input:
% NPTS     - Vector containing the number of points per dimension of
%            the discretization
% [BCS]    - Boundary conditions for each variable.  0 [default]
%            gives dirchlet (to a point not included in our grid).  1
%            gives periodic bc's in that variable.
% Output:
% MAT      - Discretized Laplacian
% VERTICES - Location of the vertices
% DN       - Number of dirchlet neighbors per vertex (does not work
%            with Neuman bc's).
%

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


function [MAT,VERTICES,DN]=laplacianfun(NPTS,varargin)

% Check for BC's
OPTS=nargin-1;
if (OPTS >= 1) BCS=varargin{1};
else BCS=0*NPTS;end
if (OPTS == 2) SCALE=varargin{2};
else SCALE=false;end

% Pull Constants
NDIM=length(NPTS);
N=prod(NPTS);

% Sanity check
if (length(NPTS) ~= length(BCS)) fprintf('Error: Size mismatch between NPTS and BCS\n');return;end

% Compute jumps (normal)
JUMP=cumprod(NPTS);JUMP=[1,JUMP(1:NDIM)];

% Compute jumps (periodic)
JP=JUMP(2:NDIM+1)-JUMP(1:NDIM);

% Diagonal
MAT=2*NDIM*speye(N,N);

% Assembly
for I=1:NDIM,
  VEC=repmat([ones(JUMP(I)*(NPTS(I)-1),1);zeros(JUMP(I),1)],N/JUMP(I+1),1);
  VEC=VEC(1:N-JUMP(I));
  MAT=MAT-spdiags([zeros(JUMP(I),1);VEC],JUMP(I),N,N) - spdiags([VEC;zeros(JUMP(I),1)],-JUMP(I),N,N);
  if(BCS(I)==1)
    VEC=repmat([ones(JUMP(I),1);zeros(JUMP(I)*(NPTS(I)-1),1)],N/JUMP(I+1),1);
    VEC=VEC(1:N-JP(I));
    MAT=MAT-spdiags([zeros(JP(I),1);VEC],JP(I),N,N) - spdiags([VEC;zeros(JP(I),1)],-JP(I),N,N);
  end
end

% Nodal Location
VERTICES=[1:NPTS(1)]';
for I=2:NDIM,
  SZ=size(VERTICES,1);
  VERTICES=[repmat(VERTICES,NPTS(I),1),reshape(repmat(1:NPTS(I),SZ,1),SZ*NPTS(I),1)];
end

% Dirchlet Neighbors
DEG=sum(abs(MAT)>0,2);
DN=full(max(DEG)-DEG);
