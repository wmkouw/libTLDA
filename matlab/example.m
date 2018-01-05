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
% rba       robust bias-aware
% scl       structural correspondence learning
% flda      feature-level domain-adaptation
%
% Last update: 20-12-2017

close all;
clearvars;
addpath(genpath('util'));

%% Select adaptive classifier to test

aclfr = 'iw';

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
M = 40;

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
        D = 50;
        
        % Source domain
        lambda1 = linspace(1e-5,2,D);
        lambda2 = linspace(1e-5,2,D)+1;
        pi_S = [1./2, 1./2];
        X = [poissrnd(ones(round(N*pi_S(1)),1)*lambda1, [round(N*pi_S(1)), D]);
             poissrnd(ones(round(N*pi_S(2)),1)*lambda2, [round(N*pi_S(2)), D])];
        y = [1*ones(round(N*pi_S(1)),1);
             2*ones(round(N*pi_S(2)),1)];
        
        % Target domain
        pi_T = [1./2, 1./2];
        lambda1 = linspace(1e-5,1,D);
        lambda2 = linspace(1,2,D);
        Z = [poissrnd(ones(round(M*pi_T(1)),1)*lambda1, [round(M*pi_T(1)), D]);
             poissrnd(ones(round(M*pi_T(2)),1)*lambda2, [round(M*pi_T(2)), D])];
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
        [~,pred] = suba(X,Z,y, 'nC', floor(D/2));
    case 'tca'
        [~,pred] = tca(X,Z,y, 'nC', 2);
    case 'flda'
        [~,pred] = flda(X,Z,y, 'loss', 'lr', 'l2', 0);
    case 'gfk'
        [~,pred] = gfk(X,Z,y, 'd', floor(D/2));
    case 'scl'
        [~,pred] = scl(X,Z,y, 'm', 50, 'h', 25);
    case 'rba'
        [~,pred] = rba(X,Z,y, 'gamma', 0.1);
    otherwise
        error('Classifier unknown');
end

% Error rate
err_adapt = mean(pred~=u);

%% Report results
disp(['Error naive: ' num2str(err_naive)]);
disp(['Error adapt: ' num2str(err_adapt)]);
