function [err,pred] = lda_err(X,y,mu,La)
% Function to compute LDA error rate

% Classes
[N,D] = size(X);
uy = unique(y);
K = numel(uy);

% Check for precomputation of svd of precision matrix
if iscell(La);
    U = La{1};
    S = La{2};
    V = La{3};
else
    [U,S,V] = svd(La);
end

% Log partition function constant
C = (-D*log(2*pi)+sum(log(diag(S)),1))./2;

% Initialize log-likelihood with partition function constant
pk = zeros(N,K);
for k = 1:K
    % Compute log-likelihood of an unlabeled sample for class k
    pk(:,k) = C - 1./2*sum((bsxfun(@minus,X,mu(k,:))*(U*sqrt(S)*V')).^2,2);
end

% Compute mean of misclassified objects
[~,pred] = max(pk, [], 2);
err = mean(pred~=y);


end
