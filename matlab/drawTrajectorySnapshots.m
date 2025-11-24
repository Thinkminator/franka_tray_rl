function drawTrajectorySnapshots(robot, theta, opts)
% drawTrajectorySnapshots  Draw multiple poses along a trajectory with tray axes and object box
%   drawTrajectorySnapshots(robot, theta)                     -> uses defaults
%   drawTrajectorySnapshots(robot, theta, opts)               -> opts is a struct with optional fields
%
% Optional opts fields (defaults shown):
%   trayName   = 'tray_object'
%   nSnapshots = size(theta,2)
%   arrowScale = 0.06
%   linkLineWidth = 2
%   boxSize    = [0.04 0.04 0.02]
%   boxOffset  = []   % defaults to [0 0 boxSize(3)/2]
%   cmap       = []   % uses jet(nSnapshots) if empty
%   axisLimits = []   % automatic if empty
%
% Example:
%   opts = struct('trayName','tray_object','nSnapshots',10,'arrowScale',0.05);
%   drawTrajectorySnapshots(robot, theta, opts);

    % ---- input handling and defaults ----
    if nargin < 3 || isempty(opts)
        opts = struct();
    end
    if ~isstruct(opts)
        error('drawTrajectorySnapshots:BadOpts', 'opts must be a struct or omitted.');
    end

    % helper to fetch field with default
    getopt = @(s,f,d) (isfield(s,f) && ~isempty(s.(f))) * 1 .* 0 + 0; %# dummy to satisfy syntax highlighting
    % implement actual get-or-default:
    function v = g(f, d)
        if isfield(opts, f) && ~isempty(opts.(f))
            v = opts.(f);
        else
            v = d;
        end
    end

    trayName = char(g('trayName', 'tray_object'));
    nDiscretize = size(theta,2);
    nSnapshots = min(g('nSnapshots', nDiscretize), nDiscretize);
    arrowScale = g('arrowScale', 0.06);
    linkLineWidth = g('linkLineWidth', 2);
    boxSize = g('boxSize', [0.04 0.04 0.02]);
    boxOffset = g('boxOffset', []);
    cmap_in = g('cmap', []);
    axisLimits = g('axisLimits', []);

    if isempty(boxOffset)
        boxOffset = [0, 0, boxSize(3)/2];
    end
    if isempty(cmap_in)
        cmap = jet(nSnapshots);
    else
        cmap = cmap_in;
    end

    % choose snapshot indices
    if nSnapshots == nDiscretize
        idx = 1:nDiscretize;
    else
        idx = round(linspace(1, nDiscretize, nSnapshots));
    end

    % create figure
    figure('Name','Trajectory snapshots with tray axes','NumberTitle','off','Color','w','Units','normalized','Position',[0.05 0.05 0.9 0.8]);
    ax = axes('Projection','perspective');
    hold(ax,'on');
    grid(ax,'on');
    view(3);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal;

    eePath = nan(3, numel(idx));

    for ii = 1:numel(idx)
        i = idx(ii);

        % joint world positions using your helper
        X = updateJointsWorldPosition(robot, theta(:,i)); % returns nJoints x 4
        jointPos = X(:,1:3);

        % plot links
        plot3(ax, jointPos(:,1), jointPos(:,2), jointPos(:,3), '-', ...
            'Color', cmap(ii,:), 'LineWidth', linkLineWidth, 'Marker', 'o', 'MarkerSize', 3, ...
            'MarkerFaceColor', cmap(ii,:), 'MarkerEdgeColor', cmap(ii,:));

        eePath(:,ii) = jointPos(end,:).';

        % tray transform
        try
            T_tray = getTransform(robot, theta(:,i), trayName);
        catch
            cfg = homeConfiguration(robot);
            for jj = 1:numel(cfg)
                cfg(jj).JointPosition = theta(jj,i);
            end
            T_tray = getTransform(robot, cfg, trayName);
        end

        origin = T_tray(1:3,4);
        Rtr = T_tray(1:3,1:3);
        xaxis = Rtr(:,1); yaxis = Rtr(:,2); zaxis = Rtr(:,3);

        % draw axes arrows
        quiver3(ax, origin(1), origin(2), origin(3), arrowScale*xaxis(1), arrowScale*xaxis(2), arrowScale*xaxis(3), ...
            'Color',[0 0.6 0], 'LineWidth',1.5, 'MaxHeadSize', 1);
        quiver3(ax, origin(1), origin(2), origin(3), arrowScale*yaxis(1), arrowScale*yaxis(2), arrowScale*yaxis(3), ...
            'Color',[0 0 1], 'LineWidth',1.5, 'MaxHeadSize', 1);
        quiver3(ax, origin(1), origin(2), origin(3), arrowScale*zaxis(1), arrowScale*zaxis(2), arrowScale*zaxis(3), ...
            'Color',[0.8 0 0.8], 'LineWidth',1.5, 'MaxHeadSize', 1);

        % draw box (object) on tray
        bx = boxSize(1)/2; by = boxSize(2)/2; bz = boxSize(3)/2;
        corners_local = [
            -bx, -by, -bz;
             bx, -by, -bz;
             bx,  by, -bz;
            -bx,  by, -bz;
            -bx, -by,  bz;
             bx, -by,  bz;
             bx,  by,  bz;
            -bx,  by,  bz;
        ];
        corners_local = corners_local + boxOffset;
        corners_world = (Rtr * corners_local.' + origin).';
        faces = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];
        patch('Vertices',corners_world,'Faces',faces,'FaceColor',0.9*cmap(ii,:),'FaceAlpha',0.9,'EdgeColor','k','Parent',ax);
    end

    % plot end-effector path
    plot3(ax, eePath(1,:), eePath(2,:), eePath(3,:), '-k', 'LineWidth', 2);

    if ~isempty(axisLimits)
        axis(axisLimits);
    else
        axis tight;
        axpad = 0.1;
        xl = xlim(ax); yl = ylim(ax); zl = zlim(ax);
        xlim(ax, xl + [-axpad axpad]); ylim(ax, yl + [-axpad axpad]); zlim(ax, zl + [-axpad axpad]);
    end

    title('Trajectory snapshots with tray axes and object box');
    hold(ax,'off');
end