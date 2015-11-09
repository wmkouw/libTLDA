function [W,theta] = flda_qd_dropout(XQ,XP,y,lambda)
% Function to calculate feature optimization regularized least squares classifier
% Note: no optimization is necessary for ls

% Shape
[M,N] = size(XQ);

% Number of classes
K = numel(unique(y));

% Check for vector y
if size(y,1)~=N; y = y'; end

if K==2
    
    % Check for y in {-1,+1}
    if ~isempty(setdiff(unique(y), [-1,1]));
        y(y~=1) = -1;
    end
    
    % Estimate dropout transfer parameters
    theta = est_transfer_params_drop(XQ,XP);

    % First two moments of transfer distribution
    EX = bsxfun(@times, (1-theta), XQ);
    VX = diag(theta.*(1-theta)).*(XQ*XQ');

    % Least squares solution
    W = (EX*EX' + VX + lambda*eye(size(XQ,1)))\EX*y;
    W = [W -W];
else
    
    W = zeros(M,K);
    theta = cell(1,K);
    for k = 1:K
        
        % Labels
        yk = (y==k);
        
        % 50up-50down resampling
        ix = randsample(find(yk==0), floor(.5*sum(1-yk)));
        Xk = [XQ(:,ix) repmat(XQ(:,yk), [1 floor((K-1)/2)])];
        yk = [double(yk(ix)); ones(1,floor((K-1)./2)*sum(yk))'];
        yk(yk==0) = -1;
        
        % Estimate dropout transfer parameters
        theta{k} = est_transfer_params_drop(Xk,XP);
        
        % First two moments of transfer distribution
        EX = bsxfun(@times, (1-theta{k}), Xk);
        VX = diag(theta{k}.*(1-theta{k})).*(Xk*Xk');

        % Least squares solution
        W(:,k) = (EX*EX' + VX + lambda*eye(size(Xk,1)))\EX*yk;
        
    end

end

end

function [theta] = est_transfer_params_drop(XQ,XP)
% Function to estimate the parameters of a dropout transfer distribution

theta = max(0,1-mean(XP>0,2)./mean(XQ>0,2));

end
