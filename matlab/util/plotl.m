function [varargout] = plotl(w,cax,varargin)
% Function to plot a linear classifier in a given figure

% Parse plot properties
p = inputParser;
addOptional(p,'Color','k');
addOptional(p,'LineWidth',3);
addOptional(p,'LineStyle','-');
addOptional(p,'Legend',[]);
addOptional(p,'Title', '');
parse(p,varargin{:});

% Get axis lims
xl = get(gca, 'XLim');
yl = get(gca, 'YLim');

% Generate axes gridlines
x = linspace(xl(1),xl(2),101);
y = linspace(yl(1),yl(2),101);

% Generate 2D grid
[X,Y] = meshgrid(x,y);

% Compute predictions
Z = reshape(X(:)*w(1) + Y(:)*w(2) + w(3), [101, 101]);

% Decision boundary is when classifier says 0
v = [0,0];

% Compare prediction height to threshold
[~,fh] = contour(cax, X,Y,Z, v);

% Set line props
set(fh, 'LineWidth', p.Results.LineWidth, 'LineColor', p.Results.Color, 'LineStyle', p.Results.LineStyle);
if ~isempty(p.Results.Title); title(p.Results.Title); end

if nargout > 0
    varargout{1} = fh;
end

end
