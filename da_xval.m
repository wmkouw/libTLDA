function [W,Theta,err,mis,lambda] = da_xval(clf,X,yX,Z,yZ,varargin)
% Function to do crossvalidation using a domain adaptive classifier
% Assumes MxN data

% Parse hyperparameters
p = inputParser;
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 1);
addOptional(p, 'l2', [0 .01 .1 1 10]);
addOptional(p, 'nE', 100);
addOptional(p, 'm', 20);
addOptional(p, 'h', 15);
addOptional(p, 'si', 1);
parse(p, varargin{:});

% Really don't like long variable names...
nR = p.Results.nR;
nF = p.Results.nF;
nE = p.Results.nE;
m = p.Results.m;
h = p.Results.h;
si = p.Results.si;
l2 = p.Results.l2;
nL = length(l2);

% Add bias if necessary
if ~all(X(end,:)==1); X = [X; ones(1,size(X,2))]; end
if ~all(Z(end,:)==1); Z = [Z; ones(1,size(Z,2))]; end

% Shape
[~,NX] = size(X);
[~,NZ] = size(Z);

% Preallocation
W = cell(nR,nF,nL);
Theta = cell(nR,nF,nL);
mis = inf*ones(nR,NX,nL);

% Loop through l2 regularization lambda values
if ~((nL==1) && (nR==1) && (nF==1))
    for l = 1:nL
        % Repeat
        for r = 1:nR
            
            % Permute and create folds
            folds = ceil(randperm(NX)./ (NX./nF));
            
            % Loop through folds
            for f = 1:nF
                
                % Update progress
                disp(['Fold ' num2str(f) '/' num2str(nF) ' of repeat ' num2str(r) '/' num2str(nR)]);
                
                % Split off validation set
                val_X = X(:,folds==f);
                val_y = yX(folds==f);
                
                % Split off training set
                if nF==1
                    trn_X = val_X;
                    trn_y = val_y;
                else
                    trn_X = X(:,folds~=f);
                    trn_y = yX(folds~=f);
                end
                
                switch clf
                    case {'qd','tqd'}
                        % Train 1 vs. all Fisher classifier
                        W{r,f,l} = qd_1vall(trn_X,trn_y,l2(l));
                        Theta{r,f,l} = [];
                        
                        % Make predictions on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case {'lr','tlr'}
                        % Train a multiclass logistic regressor
                        W{r,f,l} = mLR(trn_X,trn_y,l2(l));
                        Theta{r,f,l} = [];
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'fir_log_d'
                        % Train fir logistic with dropout
                        [W{r,f,l},Theta{r,f,l}] = fir_log_dropout(trn_X,Z,trn_y',l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'fir_log_b'
                        % Train fir logistic with blankout
                        [W{r,f,l},Theta{r,f,l}] = fir_log_blankout(trn_X,Z,trn_y',l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'fir_qd_d'
                        % Train fir quadratic with dropout
                        [W{r,f,l}, Theta{r,f,l}] = fir_qd_dropout(trn_X,Z,trn_y',l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'fir_qd_b'
                        % Train fir quadratic with blankout
                        [W{r,f,l}, Theta{r,f,l}] = fir_qd_blankout(trn_X,Z,trn_y',l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'scl'
                        % Train a structure correspondence learner
                        [W{r,f,l},Theta{r,f,l},~] = scl(trn_X(1:end-1,:),Z(1:end-1,:),trn_y,l2(l), 'm', p.Results.m, 'h', p.Results.h);
                        val_X = [val_X(1:end-1,:); Theta{r,f,l}*val_X(1:end-1,:); ones(1,size(val_X,2))];
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'irw_kmm'
                        % Train an instance reweighted lr with kmm weights
                        [Theta{r,f,l}] = irw_est_kmm(trn_X,Z, 'theta', si, 'kernel', 'rbf');
                        W{r,f,l} = mWLR(trn_X(1:end-1,:),trn_y,Theta{r,f,l},'l2', l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'irw_log'
                        % Train an instance reweighted lr with log weights
                        Theta{r,f,l} = irw_est_log(trn_X,Z, .1);
                        W{r,f,l} = mWLR(trn_X(1:end-1,:),trn_y,Theta{r,f,l},'l2', l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'sa'
                        % Train a subspace aligned classifier
                        [W{r,f,l},~,Theta{r,f,l}] = subalign(trn_X,Z,trn_y, 'l2', l2(l), 'nE', nE);
                        val_X = [(val_X'*Theta{r,f,l})'; ones(1,size(val_X,2))];
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'tca'
                        pred = NaN(size(val_y'));
                        
                    case 'gfk'
                        pred = NaN(size(val_y'));
                        
                    otherwise
                        disp(['No crossvalidation']);
                end
                
                % Check predictions
                mis(r,folds==f,l) = (pred ~= val_y');
            end
        end
    end
end

% Select optimal regularization parameter
[~,ix] = min(mean(mean(mis,2), 1));
lambda = l2(ix);

% Train on full source set using optimal
err = zeros(1,nR);
for r = 1:nR
    switch clf
        case 'tqd'
            % Target quadratic's error is the optimal crossvalidated error
            err(r) = mean(mean(mis(:,:,ix),2),1);
        case 'tlr'
            % Target logistic's error is the optimal crossvalidated error
            err(r) = mean(mean(mis(:,:,ix),2),1);
        case 'qd'
            % 1 vs all Fisher classifier
            W = qd_1vall(X,yX,l2(ix));
            Theta = [];
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
        case 'lr'
            % Multiclass logistic regressor
            W = mLR(X,yX,l2(ix));
            Theta = [];
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
        case 'irw_log'
            % Instance reweighted lr with logistic discrimination weights
            Theta = irw_est_log(X,Z,.1);
            W = mWLR(X(1:end-1,:),yX,Theta,'l2', l2(ix));
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
        case 'irw_kmm'
            % Instance reweighted lr with kernel mean matched weights
            [Theta] = irw_est_kmm(X,Z, 'theta', si, 'kernel', 'rbf');
            W = mWLR(X(1:end-1,:),yX,Theta,'l2', l2(ix));
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
        case 'scl'
            [W,Theta,~] = scl(X(1:end-1,:),Z(1:end-1,:),yX, 'l2', l2(ix), 'm', m, 'h' ,h);
            [~,pred] = max(W'*[Z(1:end-1,:); Theta*Z(1:end-1,:); ones(1,NZ)], [], 1);
            err(r) = mean(pred ~= yZ');
        case 'sa'
            % Subspace aligned lr
            [W,~,Theta] = subalign(X,Z,yX, 'l2', l2(ix), 'nE', nE);
            [~,pred] = max(W'*[(Z'*Theta)'; ones(1,NZ)], [], 1);
            err(r) = mean(pred ~= yZ');
        case 'gfk'
            % Geodesic flow kernel with 1-NN prediction
            [pred,Theta] = gfk(X(1:end-1,:), Z(1:end-1,:), yX, 'clf', 'log');
            W = [];
            err(r) = mean(pred ~= yZ');
        case 'tca'
            % Transfer component analysis classifier
            [W,Theta,pred,err(r)] = tca(X(1:end-1,:), Z(1:end-1,:), yX, yZ, 'm', nE);
        case 'fir_qd_d'
            % fir quadratic with dropout
            [W, Theta] = fir_qd_dropout(X,Z,yX, l2(ix));
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
        case 'fir_qd_b'
            % fir quadratic with blankout
            [W,Theta] = fir_qd_blankout(X,Z,yX', l2(ix));
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
        case 'fir_log_d'
            % fir logistic with dropout
            [W,Theta] = fir_log_dropout(X,Z,yX', l2(ix));
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
        case 'fir_log_b'
            % fir logistic with blankout
            [W,Theta] = fir_log_blankout(X,Z,yX',l2(ix));
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
        otherwise
            error(['Classifier not found']);
    end
end

err = mean(err);

end