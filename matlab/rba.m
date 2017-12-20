function [theta,varargout] = rba(X,y,Z,varargin)
% Implementation of a Robust Bias-Aware classifier
%
% Reference: Liu & Ziebart (20140. Robust Classification under Sample Selection Bias. NIPS.
%
% Copyright: Wouter M. Kouw
% Last update: 19-12-2017

% Parse hyperparameters
p = inputParser;
addOptional(p, 'lambda', 1e-3);
addOptional(p, 'gamma', 1);
addOptional(p, 'order', 'first');
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'yZ', []);
addOptional(p, 'clip', 100);
parse(p, varargin{:});

% Shapes
[N,D] = size(X);
[M,~] = size(Z);
labels = unique(y);
K = numel(labels);
if K>2; error('2-class only'); end

% Feature function
switch p.Results.order
    case 'first'

        % Sufficient statistics
        fxy = zeros(N,D+1,K);
        fzy = zeros(M,D+1,K);
        for k = 1:K
            fxy(y==labels(k),:,k) = [X(y==labels(k),:) ones(sum(y==labels(k)),1)];
            fzy(:,:,k) = [Z ones(M,1)];
        end

    case 'second'
    case 'third'
    otherwise
        error('Higher-order moments than third not implemented');
end

% Compute moment-matching constraint
c = squeeze(mean(fxy,1))';

% Calculate importance weights
iw = min(p.Results.clip, 1./iw_gauss(X',Z', 'order', 'ZX', 'clip', p.Results.clip))';

% Initialize
theta = rand(K,D+1);
for n = 1:p.Results.maxIter

    % Calculate posteriors
    psi = zeros(N,K);
    for k = 1:K
        for i = 1:N
            psi(i,k) = iw(i).* theta(k,:) * fxy(i,:,k)';
        end
    end

    py = zeros(N,K);
    dL = zeros(K,D+1);
    for k = 1:K
        a = max(psi,[],2);
        py(:,k) = exp(psi(:,k)-a)./ sum(exp(bsxfun(@minus,psi,a)),2);
        for i = 1:N
            dL(k,:) = dL(k,:) + sum(py(i,k)'*fxy(i,:,:),3);
        end
    end
    dL = dL./N;

    % Compute gradient with moment-matching gradients and regularization
    dC = c - dL - p.Results.lambda.*2.*theta;
    if any(isnan(dC)); error('Numerical explosion'); end

    % Update theta
    theta = theta + dC./(n*p.Results.gamma);

    % Break or update
    if norm(dC) <= p.Results.xTol; disp(['Broke at ' num2str(n)]); break; end
    if 1; disp(['Iteration ' num2str(n) '/' num2str(p.Results.maxIter) ' - Gradient: ' num2str(norm(dC))]); end

end

% Output importance weights
varargout{1} = iw;

% Evaluate with target labels
if ~isempty(p.Results.yZ);

    % Error on target set
    post = zeros(M,K);
    for i = 1:M
        for k = 1:K
            post(i,k) = exp(theta(k,:)*fzy(i,:,k)')./sum(exp(sum(theta.*squeeze(fzy(i,:,:))',2)),1);
        end
    end
    [~,pred] = max(post, [], 2);
    varargout{2} = mean(pred~=p.Results.yZ);
    varargout{3} = pred;

end


end
