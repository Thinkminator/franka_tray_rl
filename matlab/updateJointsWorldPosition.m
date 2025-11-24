function [X, T] = updateJointsWorldPosition(robot_struct, theta)
% updateJointsWorldPosition  Forward kinematics for each joint frame.
% INPUT:
%   robot_struct - robotics.RigidBodyTree
%   theta        - joint angles vector of size [nJoints x 1] (numeric)
% OUTPUT:
%   X - nJoints x 4 matrix, each row is [x y z 1] world position of joint frame
%   T - 1 x nJoints cell array of 4x4 transforms (world <- joint frame)
%
% This routine accepts robots that use struct configs or numeric configs
% depending on robot_struct.DataFormat.

    % Ensure theta is a column vector
    theta = theta(:);
    nJoints = numel(theta);

    % Prepare outputs
    T = cell(1, nJoints);
    X = zeros(nJoints, 4);

    % Get robot bodies
    bodies = robot_struct.Bodies;
    nBodies = numel(bodies);
    isCellBodies = iscell(bodies);

    % Determine whether the robot expects struct configs or numeric configs
    % Try getting a home configuration and check its type
    hc = homeConfiguration(robot_struct);

    if isstruct(hc)
        % WORKFLOW 1: struct-array config expected
        % Start from home configuration struct array and update JointPosition fields
        tConfiguration = hc; % struct array
        if numel(tConfiguration) < nJoints
            error('updateJointsWorldPosition:NumJointsMismatch', ...
                'homeConfiguration length (%d) < theta length (%d).', numel(tConfiguration), nJoints);
        end

        for i = 1:nJoints
            tConfiguration(i).JointPosition = theta(i);
        end

        % Now compute transforms for each joint by mapping to corresponding body
        for k = 1:nJoints
            jointName = tConfiguration(k).JointName;

            % Find the body index whose Joint.Name matches jointName
            idx = [];
            for bi = 1:nBodies
                if isCellBodies
                    curBody = bodies{bi};
                else
                    curBody = bodies(bi);
                end
                if strcmp(curBody.Joint.Name, jointName)
                    idx = bi;
                    break;
                end
            end

            if isempty(idx)
                error('updateJointsWorldPosition:BodyNotFound', ...
                    'Could not find body for joint "%s".', jointName);
            end

            if isCellBodies
                bodyName = bodies{idx}.Name;
            else
                bodyName = bodies(idx).Name;
            end

            % Compute transform from base to the body frame
            T{k} = getTransform(robot_struct, tConfiguration, bodyName, robot_struct.BaseName);

            p = tform2trvec(T{k});    % 1x3 [x y z]
            X(k, :) = [p, 1];
        end

    else
        % WORKFLOW 2: numeric configuration expected (DataFormat = 'row' or 'column')
        % Select numeric vector shape consistent with robot.DataFormat
        if isprop(robot_struct, 'DataFormat')
            df = robot_struct.DataFormat;
        else
            % default to column if unknown
            df = 'column';
        end

        if strcmp(df, 'column')
            q = theta(:);       % nJoints x 1
        else
            q = theta(:)';      % 1 x nJoints
        end

        % Now compute transforms for each joint by finding the body attached to the joint index
        % We need to map joint index k to a body name. We'll use the hc variable to obtain joint names
        % If hc didn't provide names, fallback to bodies order (skip base)
        jointNames = {};
        if exist('hc','var') && ~isempty(hc)
            try
                jointNames = {hc.JointName};
            catch
                jointNames = {};
            end
        end

        % If jointNames not available, deduce joint names from the bodies array:
        if isempty(jointNames)
            jointNames = cell(1, nJoints);
            jcount = 0;
            for bi = 1:nBodies
                if isCellBodies
                    curBody = bodies{bi};
                else
                    curBody = bodies(bi);
                end
                % exclude fixed joints that are not actuated if needed
                jname = curBody.Joint.Name;
                % Skip 'fixed' or '' if no joint names, but still fill sequentially up to nJoints
                jcount = jcount + 1;
                jointNames{jcount} = jname;
                if jcount == nJoints
                    break;
                end
            end
        end

        for k = 1:nJoints
            jointName = jointNames{k};

            % Find the body index whose Joint.Name matches jointName
            idx = [];
            for bi = 1:nBodies
                if isCellBodies
                    curBody = bodies{bi};
                else
                    curBody = bodies(bi);
                end
                if strcmp(curBody.Joint.Name, jointName)
                    idx = bi;
                    break;
                end
            end

            if isempty(idx)
                error('updateJointsWorldPosition:BodyNotFoundNumeric', ...
                    'Could not find body for joint "%s".', jointName);
            end

            if isCellBodies
                bodyName = bodies{idx}.Name;
            else
                bodyName = bodies(idx).Name;
            end

            % use numeric q as configuration
            T{k} = getTransform(robot_struct, q, bodyName, robot_struct.BaseName);
            p = tform2trvec(T{k});
            X(k, :) = [p, 1];
        end
    end
end