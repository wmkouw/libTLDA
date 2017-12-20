% Script as an example run of the classifiers
close all;
clearvars;

%% Generate domains

% Dimensionality
D = 2;

% Classes 
labels = [1,2];
K = length(labels);

% Source domain
pi_S = [1./2, 1./2];
N = 100;
X = [bsxfun(@plus, randn(round(N*pi_S(1)),D), [-1 0]);
     bsxfun(@plus, randn(round(N*pi_S(2)),D), [+1 0])];
y = [1*ones(round(N*pi_S(1)),1);
     2*ones(round(N*pi_S(2)),1)];

% Target domain
pi_T = [1./2, 1./2];
M = 50;
Z = [bsxfun(@plus, randn(round(M*pi_T(1)),D), [-0.5 1]);
     bsxfun(@plus, randn(round(M*pi_T(2)),D), [+1.5 1])];
u = [1*ones(round(M*pi_T(1)),1);
     2*ones(round(M*pi_T(2)),1)];

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
W = mLR(X,y, 'l2', 0);

% Plot decision boundary
plotl(W(:,2),cax1, 'LineStyle', '--');
plotl(W(:,2),cax2, 'LineStyle', '--');

%% Call adaptive classifier

clfr = 'iw';

switch clfr
    case 'iw'
        [W,iw] = iw(X,Z,y, 'loss', 'qd', 'iwe', 'lr');        
    case 'gfk'
        W = gfk(X,Z,y);
    case 'tca'
        W = tca(X,Z,y);
    case 'flda'
        W = flda(X,Z,y);
    case 'scl'
        W = scl(X,Z,y);
    case 'rcsa'
        W = rcsa(X,Z,y);
    case 'rba'
        W = rba(X,Z,y);
    case 'l_svma'
        W = l_svma(X,Z,y);
    otherwise
        error('Classifier unknown');
end    
    
%% Visualize classifier and predictions

plotl(W(:,1), cax1);
plotl(W(:,1), cax2);
