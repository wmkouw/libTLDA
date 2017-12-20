function [pred,G] = gfk(X,Z,yX,varargin)
% Implementation of Geodesic Flow Kernel classifier
%
% Reference: Geodesic Flow Kernel for Unsupervised Domain Adaptation. Gong, et al. (2008). CVPR.
%
% Copyright: Wouter M. Kouw
% Last update: 19-12-2017

% Check sizes
[MX,NX] = size(X);
[MZ,NZ]=  size(Z);
K = max(yX);

% Parse input
p = inputParser;
addParameter(p, 'd', max(1,min(floor(min(MZ,NZ)./2)-1,100)));
addParameter(p, 'l2', 1e-3);
addParameter(p, 'clf', '1-nn')
parse(p, varargin{:});

% Prep data
% X = da_prep(X, {'sum_samp'});
% Z = da_prep(Z, {'sum_samp'});

% Find principal components
[PX,~] = eigs(X*X',MX);
[PZ,~] = eigs(Z*Z',p.Results.d);

% Find geodesic flow kernel
G = GFK([PX,null(PX')], PZ);

% Perform classification
switch p.Results.clf
    case {'1nn', '1-nn'}
        [pred] = kknn(G, X', yX', Z');
    case {'lr', 'log'}
        options.Display = 'final';
        W = minFunc(@mLR_grad, zeros((MX+1)*K,1), options, G*X, yX, p.Results.l2);
        W = [reshape(W(1:end-K), [MX K]); W(end-K+1:end)'];
        [~,pred] = max(W'*[(G*Z); ones(1,NZ)],[],1);
end

end

function G = GFK(Q,Pt)
% Input: Q = [Ps, null(Ps')], where Ps is the source subspace, column-wise orthonormal
%        Pt: target subsapce, column-wise orthonormal, D-by-d, d < 0.5*D
% Output: G = \int_{0}^1 \Phi(t)\Phi(t)' dt

% ref: Geodesic Flow Kernel for Unsupervised Domain Adaptation.
% B. Gong, Y. Shi, F. Sha, and K. Grauman.
% Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Providence, RI, June 2012.

% Contact: Boqing Gong (boqinggo@usc.edu)

N = size(Q,2); %
dim = size(Pt,2);

% compute the principal angles
QPt = Q' * Pt;
[V1,V2,V,Gam,Sig] = gsvd(QPt(1:dim,:), QPt(dim+1:end,:));
V2 = -V2;
theta = real(acos(diag(Gam))); % theta is real in theory. Imaginary part is due to the computation issue.

% compute the geodesic flow kernel
eps = 1e-20;
B1 = 0.5.*diag(1+sin(2*theta)./2./max(theta,eps));
B2 = 0.5.*diag((-1+cos(2*theta))./2./max(theta,eps));
B3 = B2;
B4 = 0.5.*diag(1-sin(2*theta)./2./max(theta,eps));
G = Q * [V1, zeros(dim,N-dim); zeros(N-dim,dim), V2] ...
    * [B1,B2,zeros(dim,N-2*dim);B3,B4,zeros(dim,N-2*dim);zeros(N-2*dim,N)]...
    * [V1, zeros(dim,N-dim); zeros(N-dim,dim), V2]' * Q';

end

function [L, dL] = mLR_grad(W, X, y, lambda)
% Implementation of logistic regression
% Wouter Kouw
% 29-09-2014
% This function expects an 1xN label vector y with labels [1,..,K]

% Shape
[M,N] = size(X);
K = max(y);
W0 = reshape(W(M*K+1:end), [1 K]);
W = reshape(W(1:M*K), [M K]);

% Compute p(y|x)
WX = bsxfun(@plus, W' * X, W0');
WX = exp(bsxfun(@minus, WX, max(WX, [], 1)));
WX = bsxfun(@rdivide, WX, max(sum(WX, 1), realmin));

% Negative log-likelihood of each sample
L = 0;
for i=1:N
    L = L - log(max(WX(y(i), i), realmin));
end
L = L + lambda .* sum([W(:); W0(:)] .^ 2);

% Only compute gradient if requested
if nargout > 1

    % Compute positive part of gradient
    pos_E = zeros(M, K);
    pos_E0 = zeros(1, K);
    for k=1:K
        pos_E(:,k) = sum(X(:,y == k), 2);
        pos_E0(k) = sum(y == k);
    end

    % Compute negative part of gradient
    neg_E = X * WX';
    neg_E0 = sum(WX, 2)';

    % Compute gradient
    dL = -[pos_E(:) - neg_E(:); pos_E0(:) - neg_E0(:)] + 2 .* lambda .* [W(:); W0(:)];

end
end

function [pred] = kknn(M, Xr, Yr, Xt)

% Calculate distance according to GFK metric
dist = repmat(diag(Xr*M*Xr'),1,size(Xt,1)) ...
    + repmat(diag(Xt*M*Xt')',length(Yr),1)...
    - 2*Xr*M*Xt';

% Sort distance
[~, minIDX] = min(dist);

% Assign label of nearest neighbour
pred = Yr(minIDX);

end
