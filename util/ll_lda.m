function [ll] = ll_lda(pi_k,mu_k,La,X,y)
% Function to compute log-likelihood of samples under an LDA model

% Shapes
K = length(pi_k);
[N,D] = size(X);

% Check whether mean is transposed
if size(mu_k,2)~=D; mu_k = mu_k'; end

% Check for precomputation of svd of precision matrix
if iscell(La);
    U = La{1};
    S = La{2};
    V = La{3};
else
    [U,S,V] = svd(La);
end

% Log partition function constant
C = (-D*log(2*pi)+sum(log(realmin+diag(S)),1))./2;

% Initialize log-likelihood with partition function constant
ll = zeros(N,K);
for k = 1:K
    
    % Compute log-likelihood of an unlabeled sample for class k
    ll(:,k) = C - 1./2*sum((bsxfun(@minus,X,mu_k(k,:))*(U*sqrt(S)*V')).^2,2) + log(pi_k(k));
    
    if exist('y','var');
        
        % Weigh likelihood with label
        if isvector(y)
            % Crisp labels
            ll(:,k) = ll(:,k).*(y==k);
        else
            % Soft labels
            ll(:,k) = ll(:,k).*y(:,k);
        end
    end
end
