function [X] = preprocess(X, prep)
% Implementation of a number of preprocessing routines on a dataset
%
% Input:
%   X:      Dataset N samples by D features
%   prep:   cell array of preprocessing techniques
%
% Output:
%   X:      Preprocessed set
%
% Last update: 20-12-2017
% Author: Wouter M. Kouw

% Shape
[N,D] = size(X);

% If single prep, wrap in cell
if ischar(prep)
    prep = cellstr(prep);
end

for i = 1:length(prep)
    switch prep{i}
        case 'binarize'
            X = double(X>=0.5);
            disp(['Binarized the data (X>0.5=1, X<0.5=0)']);
        case 'minus_mean'
            X = bsxfun(@minus, X, mean(X, 1));
            disp(['Subtracted each feature`s mean']);
        case 'minus_min'
            X = bsxfun(@minus, X, min(X, [], 1));
            disp(['Subtracted each feature`s minimum value']);
        case 'norm_max'
            const = max(X,[],1);
            X = bsxfun(@rdivide, X, const);
            X(const==0,:) = 0;
            disp(['Normalized each feature by max']);
        case 'norm_sum'
            const = mean(X,1);
            X = bsxfun(@rdivide, X, const);
            X(const==0,:) = 0;
            disp(['Normalized each feature by sum']);
        case 'norm_std'
            const = std(X,0,1,'omitnan');
            X = bsxfun(@rdivide, X, const);
            X(const==0,:) = 0;
            disp(['Normalized each feature by std. dev.']);
        case 'zscore'
            X = zscore(X,[],1);
            disp(['Z-scored each feature']);
        case 'tf_idf'
            df = log(N ./ (sum(X > 0, 2) + 1));
            X = bsxfun(@times, X, df);
            disp(['Extracted tf-idf features']);
        otherwise
            error([prep{i} ' has not been implemented']);
    end
end


end
