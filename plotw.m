function plotw(varargin)
% Function to plot a linear classifier in a given figure
% Taken from prtools (but runs without prmapping)

% Parse plot properties
p = inputParser;
addRequired(p,'w');
addOptional(p,'Color','r');
addOptional(p,'Marker', 'none');
addOptional(p,'lw',3);
addOptional(p,'ls','-');
parse(p,varargin{:});

% Shape
[M,K] = size(p.Results.w);

% Define grid
n = 1000;
V = axis;
xv = linspace(V(1),V(2),n+1);
yv = linspace(V(3),V(4),n+1);
[X, Y] = meshgrid(xv,yv);

% Values of grid
D = bsxfun(@plus, [X(:),Y(:),zeros((n+1)*(n+1),K-2)]*p.Results.w(1:M-1,:), p.Results.w(M,:));
Z = reshape(D(:,1) - D(:,2),n+1,n+1);

% Plot grid
hold on
contour(xv,yv,Z,[0 0],'Color', p.Results.Color,  ...
    'LineWidth', p.Results.lw, 'LineStyle', p.Results.ls);
    
end