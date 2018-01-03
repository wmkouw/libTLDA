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
% Options for adaptive classifiers:
% iw        importance-weighting
% suba      subspace alignment
% gfk       geodesic flow kernel
% tca       transfer component analysis
% rcsa      robust covariate shift adjustment
% rba       robust bias-aware
% scl       structural correspondence learning
% flda      feature-level domain-adaptation
%
% Last update: 20-12-2017

close all;
clearvars;
addpath(genpath('util'));

%% Select adaptive classifier to test

aclfr = 'tca';

%% Generate domains

% Class-conditional distributions
switch aclfr
    case {'flda', 'scl'}
        pd = 'poisson';
    otherwise
        pd = 'normal';
end

% Number of samples
N = 100;
M = 100;

% Classes
labels = [1,2];
K = length(labels);

switch lower(pd)
    case 'normal'

        % Dimensionality
        D = 2;

        % Source domain
        pi_S = [1./2, 1./2];
        X = [bsxfun(@plus, randn(round(N*pi_S(1)),D), -1*ones(1,D));
             bsxfun(@plus, randn(round(N*pi_S(2)),D), +1*ones(1,D))];
        y = [1*ones(round(N*pi_S(1)),1);
             2*ones(round(N*pi_S(2)),1)];

        % Target domain
        pi_T = [1./2, 1./2];
        Z = [bsxfun(@plus, randn(round(M*pi_T(1)),D), -0.5*ones(1,D));
             bsxfun(@plus, randn(round(M*pi_T(2)),D), +1.5*ones(1,D))];
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

%% Train naive classifier

% Multi-class logistic regressor
Wn = mlr(X,y, 'l2', 1e-3);

% Predictions
[~,pred_n] = max([Z,ones(M,1)]*Wn,[],2);

% Error rate
err_naive = mean(pred_n~=u);

%% Train adaptive classifier

switch aclfr
    case 'iw'
        [~,pred] = iw(X,Z,y, 'iwe', 'kd');
    case 'suba'
        [~,pred] = suba(X,Z,y, 'nE', floor(D/2));
    case 'gfk'
        [~,pred] = gfk(X,Z,y, 'd', floor(D/2));
    case 'tca'
        [~,pred,C,K] = tca(X,Z,y, 'm', 2);
    case 'rcsa'
        W = rcsa(X,Z,y);
    case 'rba'
        W = rba(X,Z,y);
    case 'scl'
        [~,pred] = scl(X,Z,y, 'm', 50, 'h', 25);
    case 'flda'
        W = flda(X,Z,y);
    otherwise
        error('Classifier unknown');
end

% Error rate
err_adapt = mean(pred~=u);

%% Report results
disp(['Error naive: ' num2str(err_naive)]);
disp(['Error adapt: ' num2str(err_adapt)]);
