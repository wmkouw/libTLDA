function [pred] = kknn(X,Z,y, varargin)
% Kernel k-nearest neighbours

% Check for equal dimensions
[~,D] = size(X);
[~,E] = size(Z);
if D~=E; error('Data dimensionalities not the same in both domains'); end

% Parse hyperparameters
p = inputParser;
addOptional(p, 'K', eye(D));
addOptional(p, 'l2', 0);
parse(p, varargin{:});

% Add regularization to distance kernel
K = p.Results.K + p.Results.l2*eye(D);

% Calculate distance according to GFK metric
d = repmat(diag(X*K*X'),1,size(Z,1)) + repmat(diag(Z*K*Z')',length(y),1) - 2*X*K*Z';

% Sort distance
[~, ix] = min(d);

% Assign label of nearest neighbour
pred = y(ix);

end
