function [iw] = iwe_kd(X, Z, l2)
% Ratio of distributions as estimated by kernel density estimators
%
% Reference:

% Check for equal dimensions
[~,D] = size(X);
[~,E] = size(Z);
if D~=E
    error('Data dimensionalities not the same in both domains');
end

if D > 2
    % Target distribution
    pZ = mvksdensity(Z, X);
    
    % Source distribution
    pX = mvksdensity(X,X);
else
    % Target distribution
    pZ = ksdensity(Z, X);
    
    % Source distribution
    pX = ksdensity(X,X);
end

iw =  pZ./ (l2 + pX);

end
