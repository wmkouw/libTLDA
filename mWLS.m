function [W] = mWLS(X,y,iw,varargin)
% Implementation of multi-class weighted least squares
% Input:
%    X is in MxN format (no augmentation)
%    y is label vector in [1,..K]
%    iw is reweighting vector (1xN)
% Output:
%    W is KxM resulting classifier
%
% Wouter Kouw
% 15-09-2014

% Parse input
p = inputParser;
addParameter(p, 'l2', 1e-3);
parse(p, varargin{:});

% Shape
[M,N] = size(X);

% Augment X
X = [X; ones(1,N)];

% Check if labels are in KxN format
if any(y== 0); y(y== 0) = 2; end
if any(y==-1); y(y==-1) = 2; end
if min(size(y))==1;
    K = numel(unique(y));
    Y = zeros(K,N);
    for i = 1:N; Y(y(i),i) = 1; end
else
    Y = y;
end

% Expectation of the negative log-likelihood
bX = [bsxfun(@times, iw, X(1:M,:)); X(end,:)];

% Gradient w.r.t. w
W = (Y*bX'/ (bX*bX'+p.Results.l2*eye(M+1)))';

end
