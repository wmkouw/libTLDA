function [varargout] = plotl(w,varargin)
% Function to plot a linear classifier in a given figure

% Parse plot properties
p = inputParser;
addOptional(p,'Color','r');
addOptional(p,'Marker', 'none');
addOptional(p,'LineWidth',3);
addOptional(p,'LineStyle','-');
addOptional(p,'Legend',[]);
parse(p,varargin{:});

% Get axis lims
xl = get(gca, 'XLim');
yl = get(gca, 'YLim');

% Plot
plf = @(x1,x2) w(3) + x1*w(1) + x2*w(2);
h = ezplot(plf, [xl(1) xl(2) yl(1) yl(2)]);

% Set line props
set(h, 'LineWidth', p.Results.LineWidth, 'LineColor', p.Results.Color, 'LineStyle', p.Results.LineStyle);
title('')

if nargout==1; varargout{1} = h; end

end
