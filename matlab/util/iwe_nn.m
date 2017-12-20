function [iw] = iwe_nn(X,Z,varargin)
% Nearest-neighbours for importance weight estimation.
%
% Reference: Loog (2012), Nearest Neighbour-Based Importance Weighting. MLSP.

% Parse optionals
p = inputParser;
addOptional(p, 'clip', Inf);
addOptional(p, 'Laplace', true);
parse(p, varargin{:});

% Calculate Euclidean distance
D = pdist2(X, Z);

% Count how many target samples are in Voronoi Tesselation
[~,ix] = min(D, [], 1);
iw = hist(ix, 1:size(X,1))';

% Laplace smoothing
if p.Results.Laplace
    iw = (iw+1)./(length(iw)+1);
end

% Weight clipping
iw = min(p.Results.clip,max(0,iw));

end
