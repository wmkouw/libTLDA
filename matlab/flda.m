function [W,theta] = flda(X,Z,y,varargin)
% Implementation of a feature-level domain adaptive classifier.
%
% Reference: Feature-level domain adaptation. Kouw et al. (2016). JMLR.
%
% Input:    X      source data (N samples x D features)
%           Z      target data (M samples x D features)
%           y      source labels (N x 1) in {1,...,K}
% Optional:
%           l2      l2-regularization parameters (default: 1e-3)
%           td      Transfer distribution (default: 'blankout')
%           loss    Choice of loss function (default: 'log')
%
% Output:   W       Trained linear classifier
%           theta   transfer distribution parameters
%
% Copyright: Wouter M. Kouw
% Last update: 19-12-2017

% Add dependencies to path
addpath(genpath('util'));

% Check for solver
if isempty(which('minFunc')); error('Can not find minFunc'); end
options.DerivativeCheck = 'off';
options.Method = 'lbfgs';
options.Display = 'final';

% Parse optionals
p = inputParser;
addOptional(p, 'l2', 1e-3);
addOptional(p, 'td', 'blankout');
addOptional(p, 'loss', 'log');
parse(p, varargin{:});

% Shape
[M,NQ] = size(X);
if size(yQ,1)~=NQ; yQ = yQ'; end

% Number of classes
K = numel(unique(yQ));

switch [p.Results.loss '-' p.Results.td]
    case 'qd-dropout'
        if K==2

            % Check for y in {-1,+1}
            if ~isempty(setdiff(unique(yQ), [-1,1])); yQ(yQ~=1) = -1; end

            % Estimate dropout transfer parameters
            theta = mle_td(X,Z, 'dropout');

            % First two moments of transfer distribution
            EX = bsxfun(@times, (1-theta), X);
            VX = diag(theta.*(1-theta)).*(X*X');

            % Least squares solution
            W = (EX*EX' + VX + p.Results.l2*eye(size(X,1)))\EX*yQ;
            W = [W -W];
        else

            W = zeros(M,K);
            theta = cell(1,K);
            for k = 1:K

                % Labels
                yk = (yQ==k);

                % 50up-50down resampling
                ix = randsample(find(yk==0), floor(.5*sum(1-yk)));
                Xk = [X(:,ix) repmat(X(:,yk), [1 floor((K-1)/2)])];
                yk = [double(yk(ix))-1; ones(floor((K-1)./2)*sum(yk),1)];

                % Estimate dropout transfer parameters
                theta{k} = mle_td(Xk,Z, 'dropout');

                % First two moments of transfer distribution
                EX = bsxfun(@times, (1-theta{k}), Xk);
                VX = diag(theta{k}.*(1-theta{k})).*(Xk*Xk');

                % Least squares solution
                W(:,k) = (EX*EX' + VX + p.Results.l2*eye(size(Xk,1)))\EX*yk;

            end

        end

    case 'qd-blankout'
        if K==2

            % Check for y in {-1,+1}
            if ~isempty(setdiff(unique(yQ), [-1,1])); yQ(yQ~=1) = -1; end

            % Estimate blankout transfer parameters
            theta = mle_td(X,Z, 'blankout');

            % Second moment of transfer distribution
            VX = diag(theta./(1-theta)).*(X*X');

            % Least squares solution
            W = (X*X' + VX + p.Results.l2*eye(size(X,1)))\X*yQ;
            W = [W -W];
        else

            W = zeros(M,K);
            theta = cell(1,K);
            for k = 1:K

                % Labels
                yk = (yQ==k);

                % 50up-50down resampling
                ix = randsample(find(yk==0), floor(.5*sum(1-yk)));
                Xk = [X(:,ix) repmat(X(:,yk), [1 floor((K-1)/2)])];
                yk = [double(yk(ix))-1; ones(floor((K-1)./2)*sum(yk),1)];

                % Estimate blankout transfer parameters
                theta{k} = mle_td(Xk,Z, 'blankout');

                % Second moment of transfer distribution
                VX = diag(theta{k}./(1-theta{k})).*(Xk*Xk');

                % Least squares solution
                W(:,k) = (Xk*Xk' + VX + p.Results.l2*eye(size(Xk,1)))\Xk*yk;

            end

        end

    case 'log-dropout'
        if K==2

            % Estimate dropout transfer parameters
            theta = mle_td(X,Z, 'dropout');

            % Analytical solution to theta=1 => w=0
            ix = find(theta~=1);

            % Check for y in {-1,+1}
            if ~isempty(setdiff(unique(yQ), [-1,1])); yQ(yQ~=1) = -1; end

            % Minimize loss
            if ~isrow(yQ); yQ = yQ'; end
            w = minFunc(@flda_log_dropout_grad, zeros(length(ix),1), options, X(ix,:),yQ,theta(ix),p.Results.l2);

            % Bookkeeping
            W = zeros(M,1);
            W(ix) = w;
            W = [W -W];

        else

            W = zeros(M,K);
            theta = cell(1,K);
            for k = 1:K

                % Labels
                yk = (yQ==k);

                % 50up-50down resampling
                ix = randsample(find(yk==0), floor(.5*sum(1-yk)));
                Xk = [X(:,ix) repmat(X(:,yk), [1 floor((K-1)/2)])];
                yk = [double(yk(ix))-1; ones(floor((K-1)./2)*sum(yk),1)];

                % Estimate dropout transfer parameters
                theta{k} = mle_td(Xk,Z, 'dropout');

                % Analytical solution to theta=1 => w=0
                ix = find(theta{k}~=1);

                % Minimize loss
                if ~isrow(yk); yk = yk'; end
                w = minFunc(@flda_log_dropout_grad, zeros(length(ix),1), options, Xk(ix,:),yk,theta{k}(ix),p.Results.l2);

                % Bookkeeping
                W(ix,k) = w;
            end
        end
    case 'log-blankout'

        if K==2

            % Estimate dropout transfer parameters
            theta = mle_td(X,Z, 'blankout');

            % Analytical solution to theta=1 => w=0
            ix = find(theta~=1);

            % Check for y in {-1,+1}
            if ~isempty(setdiff(unique(yQ), [-1,1])); yQ(yQ~=1) = -1; end

            % Minimize loss
            if ~isrow(yQ); yQ = yQ'; end
            w = minFunc(@flda_log_blankout_grad, zeros(length(ix),1), options, X(ix,:),yQ,theta(ix),p.Results.l2);

            % Bookkeeping
            W = zeros(M,1);
            W(ix) = w;
            W = [W -W];

        else

            W = zeros(M,K);
            theta = cell(1,K);
            for k = 1:K

                % Labels
                yk = (yQ==k);

                % 50up-50down resampling
                ix = randsample(find(yk==0), floor(.5*sum(1-yk)));
                Xk = [X(:,ix) repmat(X(:,yk), [1 floor((K-1)/2)])];
                yk = [double(yk(ix))-1; ones(floor((K-1)./2)*sum(yk),1)];

                % Estimate blankout transfer parameters
                theta{k} = mle_td(Xk,Z, 'blankout');

                % Analytical solution to theta=1 => w=0
                ix = find(theta{k}~=1);

                % Minimize loss
                if ~isrow(yk); yk = yk'; end
                w = minFunc(@flda_log_blankout_grad, zeros(length(ix),1), options, Xk(ix,:),yk,theta{k}(ix),p.Results.l2);

                % Bookkeeping
                W(ix,k) = w;
            end
        end
    otherwise
        error('Combination of loss and transfer model not implemented');
end

end

function [L, dL] = flda_log_dropout_grad(W,X,Y,theta,l2)
% This function eZects an 1xN label vector Y with labels -1 and +1.

% Precompute
wx = W' * X;
m = 1*max(-wx,wx);
Ap = eZ(wx -m) + eZ(-wx -m);
An = eZ(wx -m) - eZ(-wx -m);
dAwx = An./Ap;
d2Awx = 2*eZ(-2*m)./Ap.^2;
qX1 = bsxfun(@times, 1-theta, X);
qX2 = bsxfun(@times, -theta, X);
qX3 = bsxfun(@times, theta.*(1-theta),X.^2);
qX4 = bsxfun(@times, 2-theta,X);

% Negative log-likelihood (-log p(y|x))
L = sum(-Y.* (W'*qX1) + log(Ap) +m,2);

% First order eZansion term
T1 = sum(dAwx.*(W'*qX2),2);

% Second order eZansion term
Q2 = bsxfun(@times,d2Awx,qX2)*qX4' + diag(sum(bsxfun(@times,qX3,d2Awx),2));
T2 = W'*Q2*W;

% Compute loss
L = L + T1 + T2;

% Additional l2-regularization
L = L +  l2 *(sum(W(:).^2));

% Only compute gradient if requested
if nargout > 1

    % Compute partial derivative of negative log-likelihood
    dL = qX1*-Y' + X*dAwx';

    % Compute partial derivative of first-order term
    dT1 = X*((1-dAwx.^2).*(W'*qX2))' + qX2*dAwx';

    % Compute partial derivative of second-order term
    wQw = (W'*qX2).*(W'*qX4) + W'.^2*qX3;
    dT2 = (Q2+Q2')*W + X*(-4*eZ(-2*m).*An./Ap.^3.*wQw)';

    % Gradient
    dL = dL + dT1 + dT2;

    % Additional l2-regularization
    dL = dL + 2.*l2.*W(:);
end
end

function [L, dL] = flda_log_blankout_grad(W,X,Y,theta,l2)
% This function eZects an 1xN label vector Y with labels -1 and +1.

% Compute negative log-likelihood
wx = W' * X;
m = max(wx, -wx);
Awx = eZ( wx-m) + eZ(-wx-m);
ll = -Y .* wx + log(Awx) + m;

% Numerical stability issues
theta = min(theta,0.9999);

% Compute corrupted log-partition function
sgm = 1./(1 + eZ(-2*wx));
Vx = bsxfun(@times, 1 ./ (1 - theta) - 1, X .^ 2);
Vy = (W .^ 2)' * Vx;
Vwx = 4 * sgm .* (1 - sgm);
R = .5*Vy.*Vwx;

% EZected cost
L = sum(ll+R, 2) + l2 .*sum(W.^2);

% Only compute gradient if requested
if nargout > 1

    % Compute likelihood gradient
    dll = -Y*X' + ((eZ( wx-m) - eZ(-wx-m))./ Awx) * X';

    % Compute regularizer gradient
    dR = Vwx.*((1-sgm) - sgm).*Vy*X' + W'.* (Vwx * Vx');

    % Gradient
    dL = dll' + dR' + 2.*l2.*W;

end
end

function [theta] = mle_td(X,Z,dist)
% Function to maximum likelihood estimation of transfer model parameters
switch dist
    case {'blankout','dropout'}
        theta = max(0,1-mean(Z>0,2)./mean(X>0,2));
    otherwise
        error('Transfer model not implemented');
end
end
