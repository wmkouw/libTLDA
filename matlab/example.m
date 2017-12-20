% libTLDA Example script to show how to run an adaptive classifier
%
% Generate synthetic data sets with either normal distributions or Poisson
% distributions (for scl and flda)
%
% X = N samples by D features data matrix for source domain
% y = N samples by 1 vector of labels in {1,...,K} for source domain
% Z = M samples by D features data matrix for target domain
% u = M samples by 1 vector of labels in {1,...,K} for target domain
%   
%
% Last update: 20-12-2017

close all;
clearvars;

addpath(genpath('util'));

%% Generate domains

% Class-conditional distributions
pd = 'poisson';

% Number of samples
N = 500;
M = 1000;

% Classes
labels = [1,2];
K = length(labels);

switch lower(pd)
    case 'normal'
        
        % Dimensionality
        D = 2;
        
        % Source domain
        pi_S = [1./2, 1./2];
        X = [bsxfun(@plus, randn(round(N*pi_S(1)),D), [-1 0]);
            bsxfun(@plus, randn(round(N*pi_S(2)),D), [+1 0])];
        y = [1*ones(round(N*pi_S(1)),1);
            2*ones(round(N*pi_S(2)),1)];
        
        % Target domain
        pi_T = [1./2, 1./2];
        Z = [bsxfun(@plus, randn(round(M*pi_T(1)),D), [-0.5 1]);
            bsxfun(@plus, randn(round(M*pi_T(2)),D), [+1.5 1])];
        u = [1*ones(round(M*pi_T(1)),1);
            2*ones(round(M*pi_T(2)),1)];
        
    case 'poisson'
        
        % Dimensionality
        D = 500;
        
        % Source domain
        pi_S = [1./2, 1./2];
        X = [poissrnd(.2*ones(round(N*pi_S(1)),D), [round(N*pi_S(1)), D]);
             poissrnd(.5*ones(round(N*pi_S(2)),D), [round(N*pi_S(2)), D])];
        y = [1*ones(round(N*pi_S(1)),1);
            2*ones(round(N*pi_S(2)),1)];
        
        % Target domain
        pi_T = [1./2, 1./2];
        Z = [poissrnd(.3*ones(round(M*pi_T(1)),D), [round(M*pi_T(1)), D]);
             poissrnd(.6*ones(round(M*pi_T(2)),D), [round(M*pi_T(2)), D])];
        u = [1*ones(round(M*pi_T(1)),1);
            2*ones(round(M*pi_T(2)),1)];
        
end

%% Visualize domains

% Determine axes limits
x_axis = [min([X(:,1); Z(:,1)]), max([X(:,1); Z(:,1)])];
y_axis = [min([X(:,2); Z(:,2)]), max([X(:,2); Z(:,2)])];

% Set figure properties
fg = figure(1);
set(gcf, 'Color', 'w', 'Position', [100 100 1200 600]);

% Scatter source domain
cax1 = subplot(1,2,1);
hold on
title('Source domain')
scatter(X(y==labels(1),1), X(y==labels(1),2), 'r', 'MarkerFaceColor', 'r');
scatter(X(y==labels(2),1), X(y==labels(2),2), 'b', 'MarkerFaceColor', 'b');
set(gca, 'XLim', x_axis, 'YLim', y_axis);

% Scatter target domain
cax2 = subplot(1,2,2);
hold on
title('Target domain')
scatter(Z(u==labels(1),1), Z(u==labels(1),2), 'r', 'MarkerFaceColor', 'r');
scatter(Z(u==labels(2),1), Z(u==labels(2),2), 'b', 'MarkerFaceColor', 'b');
set(gca, 'XLim', x_axis, 'YLim', y_axis);

%% Call naive classifier

% Multi-class logistic regressor
Wn = mLR(X,y, 'l2', 1e-3);

% Error rate
[~,pred_n] = max([Z,ones(M,1)]*Wn,[],2);
err_naive = mean(pred_n~=u);

% Plot decision boundary
plotl(Wn(:,2),cax1, 'LineStyle', '--');
plotl(Wn(:,2),cax2, 'LineStyle', '--');


%% Call adaptive classifier

clfr = 'scl';

switch clfr
    case 'iw'
        [W,iw,pred_a] = iw(X,Z,y, 'loss', 'qd', 'iwe', 'lr');
    case 'gfk'
        W = gfk(X,Z,y);
    case 'tca'
        W = tca(X,Z,y);
    case 'flda'
        W = flda(X,Z,y);
    case 'scl'
        [W,C,pred_a] = scl(X,Z,y, 'l2', 1e-6, 'm', 50, 'h', 25);
    case 'rcsa'
        W = rcsa(X,Z,y);
    case 'rba'
        W = rba(X,Z,y);
    case 'l_svma'
        W = l_svma(X,Z,y);
    otherwise
        error('Classifier unknown');
end

% Error rate
err_adapt = mean(pred_a~=u);

%% Visualize classifier and predictions

plotl(W(:,1), cax1);
plotl(W(:,1), cax2);

disp(['Error naive: ' num2str(err_naive)]);
disp(['Error adapt: ' num2str(err_adapt)]);

