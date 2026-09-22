function plotimages1(images, Y, proportion, color)
% PLOTIMAGES Plot images at 2-D coordinates with automatic size.
%
%   plotimages(images, Y)
%   plotimages(images, Y, proportion)
%   plotimages(images, Y, proportion, color)
%
% INPUTS:
%   images      : H x W x N array of images
%   Y           : 2 x N coordinates
%   proportion  : fraction of images to display (default = 1)
%   color       : RGB color, e.g.
%                 [0 0 0] = black
%                 [1 0 0] = red
%                 [0 0 1] = blue
%
% Image size is determined automatically from point spacing.
% Background is transparent.
% Orientation is preserved like the original plotimages function.


% =========================================================
% Defaults
% =========================================================

if nargin < 3 || isempty(proportion)
    proportion = 1;
end

if nargin < 4 || isempty(color)
    color = [0 0 0];
end

color = color(:)';


% =========================================================
% Check inputs
% =========================================================

n_images = size(images,3);

if size(Y,1) ~= 2
    error('Y must be a 2 x N matrix.');
end

if size(Y,2) ~= n_images
    error('Number of columns in Y must equal number of images.');
end

if proportion <= 0 || proportion > 1
    error('proportion must satisfy 0 < proportion <= 1.');
end

if numel(color) ~= 3
    error('color must be an RGB vector, for example [1 0 0].');
end


% =========================================================
% Select images
% =========================================================

nplot = max(1, round(proportion*n_images));

idx = round(linspace(1,n_images,nplot));

idx = unique(idx,'stable');


% =========================================================
% Current axes
% =========================================================

ax = gca;

drawnow;

xl = xlim(ax);
yl = ylim(ax);

xrange = diff(xl);
yrange = diff(yl);


% =========================================================
% Actual axes size in screen pixels
% =========================================================

oldUnits = ax.Units;

ax.Units = 'pixels';

p = ax.InnerPosition;

axWidthPx  = p(3);
axHeightPx = p(4);

ax.Units = oldUnits;


% =========================================================
% Convert selected Y points to screen coordinates
% =========================================================

xp = (Y(1,idx)-xl(1)) ./ xrange .* axWidthPx;
yp = (Y(2,idx)-yl(1)) ./ yrange .* axHeightPx;


% =========================================================
% Find typical nearest-neighbor distance in screen pixels
% =========================================================

m = length(idx);

if m > 1

    DX = xp(:) - xp(:)';
    DY = yp(:) - yp(:)';

    D = sqrt(DX.^2 + DY.^2);

    % Ignore distance to itself
    D(1:m+1:end) = Inf;

    nearest = min(D,[],2);

    nearest = nearest(isfinite(nearest) & nearest > 0);

    if isempty(nearest)

        spacingPx = sqrt(axWidthPx*axHeightPx/m);

    else

        spacingPx = median(nearest);

    end

else

    spacingPx = 20;

end


% =========================================================
% Automatic image size
% =========================================================

fillFactor = 0.85;

imageWidthPx = fillFactor * spacingPx;


% Avoid extremely small or large images
minImageSizePx = 7;
maxImageSizePx = 30;

imageWidthPx = max(imageWidthPx,minImageSizePx);
imageWidthPx = min(imageWidthPx,maxImageSizePx);


% =========================================================
% Image dimensions
%
% IMPORTANT:
% Original function displayed:
%
%       current_image'
%
% Therefore the displayed image is transposed.
% =========================================================

originalH = size(images,1);
originalW = size(images,2);

% After transpose:
displayH = originalW;
displayW = originalH;


% Preserve image aspect ratio
imageHeightPx = imageWidthPx * displayH/displayW;


% =========================================================
% Convert pixel dimensions into DATA coordinates
%
% This compensates for different X/Y axis scales.
% =========================================================

widthData = ...
    imageWidthPx / axWidthPx * xrange;

heightData = ...
    imageHeightPx / axHeightPx * yrange;


% =========================================================
% Keep limits fixed
% =========================================================

xlim(ax,xl);
ylim(ax,yl);

ax.XLimMode = 'manual';
ax.YLimMode = 'manual';


% =========================================================
% Hold state
% =========================================================

wasHold = ishold(ax);

hold(ax,'on');


% =========================================================
% Plot images
% =========================================================

for ii = 1:length(idx)

    k = idx(ii);


    % -----------------------------------------------------
    % Get image
    % -----------------------------------------------------

    im = double(images(:,:,k));


    % -----------------------------------------------------
    % Normalize only if necessary
    % -----------------------------------------------------

    imin = min(im(:));
    imax = max(im(:));

    if imax > imin

        im = (im-imin)/(imax-imin);

    else

        im = zeros(size(im));

    end


    % -----------------------------------------------------
    % IMPORTANT:
    %
    % Your ORIGINAL function used:
    %
    %       current_image'
    %
    % so transpose the image.
    % -----------------------------------------------------

    im = im.';


    % -----------------------------------------------------
    % Transparency
    %
    % For your data:
    %
    %       background -> approximately 0
    %       digit      -> approximately 1
    %
    % Therefore:
    %
    %       AlphaData = im
    %
    % NOT 1-im.
    %
    % Background becomes transparent.
    % -----------------------------------------------------

    mask = im;

    % Make absolutely sure alpha is between 0 and 1
    mask = max(0,min(1,mask));


    % -----------------------------------------------------
    % RGB colored image
    % -----------------------------------------------------

    rgb = zeros(size(im,1),size(im,2),3);

    rgb(:,:,1) = color(1);
    rgb(:,:,2) = color(2);
    rgb(:,:,3) = color(3);


    % -----------------------------------------------------
    % Location
    % -----------------------------------------------------

    xc = Y(1,k);
    yc = Y(2,k);


    % -----------------------------------------------------
    % Plot
    %
    % Notice YData goes from HIGH to LOW.
    %
    % This reproduces the orientation of your original:
    %
    % [ys+height  ys]
    % -----------------------------------------------------

    h = image(ax, ...
        'XData', ...
        [xc-widthData/2, xc+widthData/2], ...
        'YData', ...
        [yc+heightData/2, yc-heightData/2], ...
        'CData',rgb, ...
        'AlphaData',mask, ...
        'AlphaDataMapping','none');

end


% =========================================================
% Restore limits
% =========================================================

xlim(ax,xl);
ylim(ax,yl);


% =========================================================
% Labels
% =========================================================

xlabel(ax,'x');
ylabel(ax,'y');


% =========================================================
% Restore HOLD
% =========================================================

if ~wasHold
    hold(ax,'off');
end

end