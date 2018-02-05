function da_scatter(X,Z,y,varargin)
% Scatterplot of data in both domains

% Parse hyperparameters
p = inputParser;
addOptional(p, 'u', []);
addOptional(p, 'fh', []);
parse(p, varargin{:});

% Labels
labels = unique(y);

% Determine axes limits
x_axis = [min([X(:,1); Z(:,1)]), max([X(:,1); Z(:,1)])];
y_axis = [min([X(:,2); Z(:,2)]), max([X(:,2); Z(:,2)])];

% Set figure properties
if isempty(p.Results.fh)
    fg = figure(1); 
else
    fg = p.Results.fh;
end
set(fg, 'Color', 'w', 'Position', [100 100 1200 600]);

% Scatter source domain
cax1 = subplot(1,2,1);
set(cax1, 'XLim', x_axis, 'YLim', y_axis);
hold on

scatter(X(y==labels(1),1), X(y==labels(1),2), 'r', 'MarkerFaceColor', 'r');
scatter(X(y==labels(2),1), X(y==labels(2),2), 'b', 'MarkerFaceColor', 'b');

xlabel('x1');
ylabel('x2');
title('Source domain')

% Scatter target domain
cax2 = subplot(1,2,2);
set(cax2, 'XLim', x_axis, 'YLim', y_axis);
hold on

if isempty(p.Results.u)
    scatter(Z(:,1), Z(:,2), 'k', 'MarkerFaceColor', 'k');
else
    scatter(Z(p.Results.u==labels(1),1), Z(p.Results.u==labels(1),2), 'r', 'MarkerFaceColor', 'r');
    scatter(Z(p.Results.u==labels(2),1), Z(p.Results.u==labels(2),2), 'b', 'MarkerFaceColor', 'b');
end

xlabel('x1');
ylabel('x2');
title('Target domain')

end
