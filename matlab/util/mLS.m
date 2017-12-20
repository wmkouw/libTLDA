function [W] = mLS(X,y,varargin)
% Multi-class least-squares
%
% Input:
%    X is N samples by D features
%    y is N samples by 1 vector in [1,..K]
% Output:
%    W is D features by K classes
%
% Wouter Kouw
% 15-09-2014

% Parse input
p = inputParser;
addParameter(p, 'l2', 1e-3);
parse(p, varargin{:});

% Shape
[N,D] = size(X);

% Check for bias augmentation
if ~all(X(:,end)==1); X = [X ones(size(X,1),1)]; D = D+1; end

% Number of classes
labels = unique(y);
K = length(labels); 

% Check if labels are in [1,...,K]
if ~isempty(setdiff(labels,1:K)); error('Labels should be in [1,...,K]'); end

% Map y to one-hot encoding
Y = zeros(N,K);
for n = 1:N
    Y(n,y(n)) = 1;
end

% Closed-form solution to least-squares 
W = (X'*X + p.Results.l2*eye(N))\(X'*Y);

end
