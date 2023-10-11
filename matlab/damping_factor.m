% function df=damping_factor(A,omega,diagonal_value,exact)
%
% Input:
%  A               - Matrix A
%  omega           - Global damping parameter omega
%  diagonal_value  - Diagonal to use
%  exact           - 0=Differentiable estimate, 1="Exact" Estimate

% Computes the damping factor for weighted Jacobi on this matrix

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


function df=damping_factor(A,omega,diagonal_value,exact)
if(~exist('exact','var')), exact = 0; end
N=size(A,1);

if(size(diagonal_value,1)==1), 
  diagonal_value = reshape(diagonal_value,N,1);
end

if(exact==1),
  % "Exact" eigenvalues: Does not support automatic differentiation,
  % so this can only be used outside of the training loop
  D=spdiags(diagonal_value,0,N,N);
  J = eye(N,N) - omega * (D\A);
  opts=struct; opts.tol=1e-6;
  df = abs(eigs(J,1,'largestabs',opts));    
else
  % Differentiable eigenvalue estimate via the Power Method  
  x=ones(N,1);
  nits = 20;
  
  for I=1:nits,
    x=jacobi_dl(diagonal_value,omega,A,x);
    xnrm = sqrt(x'*x);  
    x = x/ xnrm;
  end
  
  ritz_top = x'*jacobi_dl(diagonal_value,omega,A,x);
  ritz_bottom = 1; % because we normalized x
  
  df = abs(ritz_top / ritz_bottom);
end


% Single step of weighted Jacobi iteration
function y=jacobi_dl(dvals,omega,A,x)
Ax = A*x;

DinvA = Ax ./ dvals;

y = x - omega * DinvA;


