function [theta,varargout] = tce(X,yX,Z,varargin)
% Function to run the Target Contrastive Estimator
% Input:
% 		    X      	source data (N samples x D features)
%           Z      	target data (M samples x D features)
%           yX 	   	source labels (N x 1)
% Optional input:
%     		yZ 		target labels (M samples x 1, for evaluation)
% 			model	loss or likelihood based classification model (default: 'ls')
% 			alpha 	learning rate (default: 1)
%           lambda  l2-regularization parameter (default: 0)
% 			maxIter maximum number of iterations (default: 500)
% 			xTol 	convergence criterion (default: 1e-5)
% 			viz		visualization during optimization (default: false)
% Output:
% 			theta   target model estimate
% Optional output:
%           {1}   	target likelihood of the tce estimate
% 			{2} 	target likelihood of the source esimate
% 			{3}		target error of the tce estimate
%			{4}		target predictions of the tce estimate
%			{5}		target error of the source estimate
%			{6}		target predictions of the source estimate
%
% Kouw & Loog (2016). Target Contrastive Estimator for Robust Domain 
% Adaptation, under review.
% Kouw & Loog (2016). Least-Squares Target Contrastive Estimation for
% Transfer Learning, under review.
% Last update: 01-04-2016

% Parse hyperparameters
p = inputParser;
addOptional(p, 'yZ', []);
addOptional(p, 'alpha', 1);
addOptional(p, 'lambda', 0);
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'viz', false);
parse(p, varargin{:});

% tce-compatible models
switch model
    
    case 'ls'
        % Size
        [M,D] = size(Z);
        lab = [-1 +1];
        K = numel(lab);
        
        % Augment data with bias if necessary
        if ~all(X(:,end)==1); X = [X ones(size(X,1),1)]; end
        if ~all(Z(:,end)==1); Z = [Z ones(size(Z,1),1)]; end
        
        % Reference parameter estimates
        theta.ref = svdinv(X'*X + p.Results.lambda*eye(D))*X'*yX;
        
        % Initialize
        q = ones(M,K)./K;
        Dq = zeros(M,K);
        theta.mcpl = theta.ref;
        
        disp('Starting MCPL optimization');
        ll_old = Inf;
        for n = 1:p.Results.maxIter
            
            %%% Minimization
            theta.mcpl = svdinv(Z'*Z + p.Results.lambda*eye(D))*Z'*(-q(:,1)+q(:,2));
            
            %%% Maximization
            % Compute new gradient
            for k = 1:K
                Dq(:,k) = (Z*theta.mcpl - lab(k)).^2 - (Z*theta.ref - lab(k)).^2;
            end
            
            % Apply gradient and project back onto simplex (alpha is accelerating/decelerating constant for 1/n learning rate)
            if n < (p.Results.maxIter./10);
                q = proj_splx((q + Dq.*100)')';
            else
                q = proj_splx((q + Dq./(p.Results.alpha * n))')';
            end
            
            % Visualize
            if p.Results.viz
                if rem(n,500)==2;
                    % Scatter first 2 dimensions and decision boundaries
                    cm = cool;
                    mk = {'x','o'};
                    figure(1);
                    clf(1)
                    hold on
                    for j = 1:size(Z,1)
                        [~,mky] = max(lab==p.Results.yZ(j),[],2);
                        plot(Z(j,1),Z(j,2), 'Color', cm(1+round(q(j,1)*63),:), 'Marker', mk{mky}, 'LineStyle', 'none');
                    end
                    h_m = plotl(theta.mcpl, 'Color','r', 'LineStyle', '-.');
                    h_r = plotl(theta.ref, 'Color','b','LineStyle',':');
                    
                    legend([h_m h_r], {'MCPL', 'ref'});
                    colorbar
                    colormap(cool)
                    
                    drawnow
                    pause(.1);
                end
            end
            
            % Minimax loss
            ll = 0;
            for k = 1:K
                ll = ll + sum(q(:,k).*((Z*theta.mcpl - lab(k)).^2)) - sum(q(:,k).*((Z*theta.ref - lab(k)).^2));
            end
            
            % Update or break
            dll = norm(ll_old-ll);
            if  dll < p.Results.xTol; disp(['Broke at ' num2str(n)]); break; end
            ll_old = ll;
            if rem(n,1000)==1; disp(['Iteration ' num2str(n) '/' num2str(p.Results.maxIter) ' - Minimax loss: ' num2str(ll)]); end
            
        end
        
    case 'lda'
        
        % Sizes
        [N,D] = size(X);
        M = size(Z,1);
        uy = unique(yX);
        K = numel(uy);
        
        % Reference parameter estimates
        Nk = zeros(1,K);
        pi_ref = NaN(1,K);
        mu_ref = NaN(K,D);
        S_ref = zeros(D);
        for k = 1:K
            Nk(k) = sum(yX==uy(k));
            pi_ref(k) = Nk(k)./N;
            mu_ref(k,:) = mean(X(yX==uy(k),:),1);
            S_ref = S_ref + (bsxfun(@minus,X(yX==uy(k),:), mu_ref(k,:))'*bsxfun(@minus,X(yX==uy(k),:), mu_ref(k,:)))./N;
        end
        La_ref = svdinv((S_ref+S_ref')./2);
        
        % Precompute log-likelihood of unlabeled samples under reference model
        ll_ref = ll_lda(pi_ref,mu_ref,La_ref,Z);
        
        % Initialize target posterior
        q = min(max(proj_splx(ll_ref), realmin), 1-realmin);
        
        disp('Starting MCPL optimization');
        llmm = Inf;
        for n = 1:p.Results.maxIter
            
            %%% Maximization
            pi_mcpl = NaN(1,K);
            mu_mcpl = NaN(K,D);
            S_mcpl = zeros(D,D,K);
            for k = 1:K;
                pi_mcpl(k) = sum(q(:,k),1)./M;
                mu_mcpl(k,:) = sum(bsxfun(@times, q(:,k), Z),1)./sum(q(:,k),1);
                S_mcpl(:,:,k) = sum(q(:,k),1).*mu_mcpl(k,:)'*mu_mcpl(k,:);
            end
            S_mcpl = (bsxfun(@times,sum(q,2),Z)'*Z - sum(S_mcpl,3))./M;
            
            % Perform singular value decomposition of covariance matrix
            [U_mcpl,S_mcpl,V_mcpl] = svd((S_mcpl+S_mcpl')./2);
            
            % Stable inverse
            S_mcpl(S_mcpl>0) = 1./S_mcpl(S_mcpl>0);
            
            %%%% Minimization
            
            % Compute new gradient
            ll_mcpl = ll_lda(pi_mcpl,mu_mcpl,{U_mcpl,S_mcpl,V_mcpl},Z);
            Dq = ll_mcpl - ll_ref;
            
            % Apply gradient and project back onto simplex
            q = min(max(proj_splx(q - Dq./(p.Results.alpha+n)), realmin), 1-realmin);
            
            % Visualize
            if p.Results.viz
                if rem(n,100)==1;
                    cm = cool;
                    mk = {'x','o'};
                    figure(1);
                    clf(1)
                    hold on
                    for j = 1:size(Z,1)
                        plot(Z(j,1),Z(j,2), 'Color', cm(1+round(q(j,1)*63),:), 'Marker', mk{p.Results.yZ(j)}, 'LineStyle', 'none');
                    end
                    drawnow
                    pause(.1);
                end
            end
            
            % Break or update
            llmm_ = ll_mcpl.*q;
            
            dll = norm(llmm-llmm_);
            if isnan(dll); error('Numeric error'); end
            if  dll < p.Results.xTol; disp(['Broke at ' num2str(n)]); break; end
            llmm = llmm_;
            if rem(n,50)==1; disp(['Iteration ' num2str(n) '/' num2str(p.Results.maxIter) ' - Minimax gradient: ' num2str(dll)]); end
            
        end
        
        % Output parameters
        La_mcpl = (V_mcpl*S_mcpl*U_mcpl');
        theta.mcpl = {pi_mcpl,mu_mcpl,La_mcpl};
        theta.ref = {pi_ref,mu_ref,La_ref};
        
end

% Evaluate with target labels
if ~isempty(p.Results.yZ);
    
    switch model
        case 'ls'
            
            % Loss
            varargout{1} = sum(double(p.Results.yZ==lab(1)).*(Z*theta.mcpl - lab(1)).^2 + double(p.Results.yZ==lab(2)).*(Z*theta.mcpl - lab(2)).^2);
            varargout{2} = sum(double(p.Results.yZ==lab(1)).*(Z*theta.ref - lab(1)).^2 + double(p.Results.yZ==lab(2)).*(Z*theta.ref - lab(2)).^2);
            
            % Error
            varargout{4} = sign(Z*theta.mcpl);
            varargout{3} = mean(varargout{4}~=p.Results.yZ);
            varargout{6} = sign(Z*theta.ref);
            varargout{5} = mean(varargout{6}~=p.Results.yZ);
            
            % Worst-case labeling
            varargout{7} = q;
            
        case 'lda'
            
            % Likelihood
            varargout{1} = sum(sum(ll_lda(pi_mcpl,mu_mcpl,La_mcpl,Z,p.Results.yZ),2),1);
            varargout{2} = sum(sum(ll_lda(pi_ref,mu_ref,La_ref,Z,p.Results.yZ),2),1);
            
            % Error
            [varargout{3},varargout{4}] = lda_err(Z,p.Results.yZ,theta.mcpl{2}, theta.mcpl{3});
            [varargout{5},varargout{6}] = lda_err(Z,p.Results.yZ,theta.ref{2}, theta.ref{3});
            
            % Worst-case labeling
            varargout{7} = q;
    end
end

end
