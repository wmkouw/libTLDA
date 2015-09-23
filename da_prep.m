function [X] = da_prep(X, prep)
% Function to run a number of preprocessing routines on a dataset
% Input:
%   X:      Dataset (MxN)
%   prep:   cell array of preps
% 
% Output:
%   X:      Prepped set
% 
% Author: Wouter Kouw
% Pattern Recognition & Bioinformatics group
% Delft University of Technology

[M,N] = size(X);

if ischar(prep)
    prep = cellstr(prep);
end

for i = 1:length(prep)
    switch prep{i}
        case 'binarize'
            X(X>=0.5) = 1;
            X(X<0.5) = 0;
            disp(['Binarized the data (X>0.5=1, X<0.5=0)']);
        case 'zscore'
            X = bsxfun(@minus, X, mean(X,2));
            X = bsxfun(@rdivide, X, var(X, 0, 2));
            disp(['Z-scored each feature']);
        case 'minus_min'
            X = bsxfun(@minus, X, min(X, [], 2));
            disp(['Subtracted the minimum value for each feature']);
        case 'tf-idf'
            df = log(N ./ (sum(X > 0, 2) + 1));
            X = bsxfun(@times, X, df);
            disp(['Ran tf-idf features']);
        case 'fmax'
            X = bsxfun(@rdivide, X, max(X, [], 2));
            disp(['Capped each feature at 1']);
        case 'fsum'
            X = bsxfun(@rdivide, X, sum(X, 2));
        disp(['Normalized each feature']);
        case 'norm_samp'
            X = bsxfun(@rdivide, X, sqrt(sum(X.^2,1)));
            disp(['Normalized each sample by their norm']);
        case ''
            disp(['No data preprocessing']);
        otherwise
            error([prep{i} ' has not been implemented']);
    end
end


end