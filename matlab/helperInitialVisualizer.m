% Visualize the obstacles and robot manipulator (interactive)

positions = x0(1:numJoints)';

hfig = figure('Position', [375 446 800 600]);
ax1 = axes('Parent', hfig);

% Show robot and keep subsequent plots (and show/world overlays) from being overwritten
% Use PreservePlot = true so MATLAB does not clear body frames each time
show(robot, positions(:,1), 'Parent', ax1, 'PreservePlot', true, 'Frames', 'off');
hold(ax1, 'on');

% Set initial camera view and axis limits
view(ax1, 150, 29);
axis(ax1, [-0.9 0.9  -0.7 0.8  -0.2 0.9]);
axis(ax1, 'equal');     % keep aspect ratio
axis(ax1, 'vis3d');     % prevent camera aspect changes during rotation

% Plot final position as a marker (ensure poseFinal is [x y z])
plot3(ax1, poseFinal(1), poseFinal(2), poseFinal(3), 'r.', 'MarkerSize', 20);

% Visualize collision objects (world). 'show' returns a patch handle for styling.
for i = 1:numel(world)
    [~, pObj] = show(world{i}, 'Parent', ax1);
    pObj.LineStyle = 'none';
    if isprop(pObj, 'FaceColor')
        pObj.FaceColor = [0.8, 0.4, 0.05];   % brown-ish
        pObj.FaceAlpha = 0.9;
    end
end

% Enable interactive controls
rotate3d(ax1, 'on');    % left-drag to rotate
zoom(ax1, 'on');        % mouse wheel to zoom
pan(ax1,  'on');        % pan mode (use toolbar button or call pan on/off)
cameratoolbar('Show');  % show camera toolbar (orbit/dolly/zoom)
cameratoolbar('SetMode','orbit');

% Optional: enable data cursor to inspect coordinates
dcm = datacursormode(hfig);
set(dcm, 'Enable', 'on', 'DisplayStyle', 'datatip');

% Optional: add a UI toggle button to enable/disable rotate mode
uicontrol('Style','togglebutton','String','Rotate','Value',1, ...
    'Position',[10 10 60 25], 'Callback', @(src,evt) toggleRotate(src, ax1));

% small helper for the toggle button
function toggleRotate(btn, ax)
    if btn.Value
        rotate3d(ax, 'on');
    else
        rotate3d(ax, 'off');
    end
end

% Example: programmatic slow rotation to inspect (run manually if you want)
% for a = 150:170
%     view(ax1, a, 29);
%     drawnow;
%     pause(0.02);
% end