function [Stheta, Qtheta] = stompTrajCost(robot_struct, theta, R, voxel_world)
% Compute the local trajectory cost at each discretization theta point,
% as well as the overall trajectory cost (the Qtheta)

[~, nDiscretize] = size(theta);

% Obstacle cost
qo_cost = zeros(1, nDiscretize);

% Constraint cost components (separated)
qc_orientation = zeros(1, nDiscretize);
qc_sliding     = zeros(1, nDiscretize);
qc_tipping     = zeros(1, nDiscretize);

% Define world z-axis
world_z = [0; 0; 1];

% ---------------------------
% Parameters (ensure defined)
% ---------------------------
mu = 0.6;              % friction coefficient (from MuJoCo XML)
g = 9.81;              % gravity
% If you have a known total motion time T_total, set dt = T_total/(nDiscretize-1)
dt = 0.1;              % time step between waypoints (tunable)
% Individual weights
w_obstacle    = 1.0;   % obstacle cost weight
w_orientation = 5.0;   % orientation alignment weight
w_sliding     = 1;   % sliding penalty weight
w_tipping     = 1;   % tipping penalty weight
% Limits
angAccLimit = 3;       % angular acceleration limit (rad/s^2)
% ---------------------------

% Check that 'tray_object' exists in the robot model (robust)
trayName = 'tray_object';
bodyNames = robot_struct.BodyNames;   % cell array
if ~any(strcmp(trayName, bodyNames))
    warning('stompTrajCost:TrayNotFound', ...
        'Body "%s" not found in robot. Using end-effector (last transform) instead.', trayName);
    useTray = false;
else
    useTray = true;
end

% Prepare a struct template once for fallback (if needed)
cfgTemplate = homeConfiguration(robot_struct);

% For the first waypoint (i=1)
[X, ~] = updateJointsWorldPosition(robot_struct, theta(:, 1));
[sphere_centers, radi] = stompRobotSphere(X);
vel = zeros(length(sphere_centers), 1);
qo_cost(1) = stompObstacleCost(sphere_centers, radi, voxel_world, vel);

% Get tray transform and orientation for waypoint 1
if useTray
    try
        T_tray = getTransform(robot_struct, theta(:,1), trayName);
    catch
        cfg = cfgTemplate;
        for jj = 1:numel(cfg)
            cfg(jj).JointPosition = theta(jj,1);
        end
        T_tray = getTransform(robot_struct, cfg, trayName);
    end
else
    [~, Ttmp] = updateJointsWorldPosition(robot_struct, theta(:,1));
    T_tray = Ttmp{end};
end
R_current = T_tray(1:3,1:3);
qc_orientation(1) = norm(R_current(:,3) - world_z, 2)^2;

% Store previous transform for central difference
T_prev = T_tray;

% Loop through remaining waypoints
for i = 2 : nDiscretize
    sphere_centers_prev = sphere_centers;

    % Forward kinematics at waypoint i
    [X, ~] = updateJointsWorldPosition(robot_struct, theta(:, i));
    [sphere_centers, radi] = stompRobotSphere(X);

    % Approximate speed for sphere centers
    vel = vecnorm(sphere_centers_prev - sphere_centers, 2, 2);
    qo_cost(i) = stompObstacleCost(sphere_centers, radi, voxel_world, vel);

    % Get tray transform for waypoint i
    if useTray
        try
            T_tray = getTransform(robot_struct, theta(:,i), trayName);
        catch
            cfg = cfgTemplate;
            for jj = 1:numel(cfg)
                cfg(jj).JointPosition = theta(jj,i);
            end
            T_tray = getTransform(robot_struct, cfg, trayName);
        end
    else
        [~, Ttmp] = updateJointsWorldPosition(robot_struct, theta(:, i));
        T_tray = Ttmp{end};
    end

    R_current = T_tray(1:3,1:3);
    qc_orientation(i) = norm(R_current(:,3) - world_z, 2)^2;

    % ---- Tray Object Constraints (Sliding & Tipping Prevention) ----
    if i > 1 && i < nDiscretize
        % get previous, current, next positions (world frame)
        p_prev = T_prev(1:3,4);
        p_curr = T_tray(1:3,4);

        % Get next transform for central difference
        if useTray
            try
                T_next = getTransform(robot_struct, theta(:,i+1), trayName);
            catch
                cfg = cfgTemplate;
                for jj = 1:numel(cfg)
                    cfg(jj).JointPosition = theta(jj,i+1);
                end
                T_next = getTransform(robot_struct, cfg, trayName);
            end
        else
            [~, Ttmp] = updateJointsWorldPosition(robot_struct, theta(:, i+1));
            T_next = Ttmp{end};
        end
        p_next = T_next(1:3,4);

        % Linear acceleration (central difference) in world frame
        a_world = (p_next - 2*p_curr + p_prev) / (dt^2);

        % Transform to tray frame
        a_tray = R_current' * a_world;

        % Sliding constraint: tangential acceleration <= μ*(g - normal acceleration)
        a_tangential = norm(a_tray(1:2));  % x,y in tray frame
        a_normal = a_tray(3);              % z in tray frame (positive upward)
        g_eff = max(0, g - a_normal);
        max_allowed_tangential = mu * g_eff;

        % store squared violation (0 if no violation)
        sliding_violation = max(0, a_tangential - max_allowed_tangential);
        qc_sliding(i) = sliding_violation^2;

        % Tipping constraint: angular acceleration estimate using wrapped Euler diffs
        % Get small-angle Euler representations for prev, curr, next
        % Use rotm2eul and wrap differences to handle wrapping
        try
            e_prev = rotm2eul(T_prev(1:3,1:3), 'XYZ');
            e_curr = rotm2eul(R_current, 'XYZ');
            e_next = rotm2eul(T_next(1:3,1:3), 'XYZ');
        catch
            % If rotm2eul unavailable, fallback to zeros (no angular penalty)
            e_prev = [0 0 0];
            e_curr = [0 0 0];
            e_next = [0 0 0];
        end

        % Wrap differences into [-pi, pi] to avoid discontinuities
        % Form central-difference angular accel with wrap handling
        d1 = wrapToPi(e_curr - e_prev);
        d2 = wrapToPi(e_next - e_curr);
        % approximate angular acceleration: (d2 - d1) / dt -> but we want central second diff:
        alpha = wrapToPi(e_next - 2*e_curr + e_prev) / (dt^2);

        % Ensure alpha has at least 2 components for roll/pitch check
        if numel(alpha) >= 2
            ang_violation = max(0, abs(alpha(1:2)) - angAccLimit); % roll/pitch
            qc_tipping(i) = sum(ang_violation.^2);
        else
            qc_tipping(i) = 0;
        end
    end

    % Update previous transform
    T_prev = T_tray;
end

% Combine all constraint terms with individual weights
Stheta = w_obstacle * qo_cost + ...
         w_orientation * qc_orientation + ...
         w_sliding * qc_sliding + ...
         w_tipping * qc_tipping;

% smoothness/control cost (exclude endpoints)
theta_interior = theta(:, 2:end-1);
Qtheta = sum(Stheta) + 1/2 * sum(theta_interior * R * theta_interior', "all");

end