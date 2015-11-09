function [W] = qd_1vall(X,y,lambda)
% Function to calculate feature absence regularized least squares classifier
% Note: no optimization is necessary for ls

% Shape
[M,N] = size(X);

% Number of classes
K = numel(unique(y));

if K==2
    
    % Check for y in {-1,+1}
    if ~isempty(setdiff(unique(y), [-1,1]));
        y(y~=1) = -1;
    end
    
    % Least squares solution
    w = (X*X' + lambda*eye(size(X,1)))\X*y;
    W = [w -w];
else
    
    W = zeros(M,K);
    for k = 1:K
        
        % Labels
        yk = (y==k);
        
        % 50up-50down resampling
        ix = randsample(find(yk==0), floor(.5*sum(yk==0)));
        Xk = [X(:,ix) repmat(X(:,yk), [1 floor((K-1)/2)])];
        yk = [double(yk(ix)); ones(1,floor((K-1)./2)*sum(yk))'];
        yk(yk==0) = -1;
        
        % Least squares solution
        W(:,k) = (Xk*Xk' + lambda*eye(size(Xk,1)))\Xk*yk;
        
    end

end

end
