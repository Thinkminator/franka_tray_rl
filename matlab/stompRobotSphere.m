%% Construct the sphere around the robot maniputor
% INPUT:
%    X: joints (x,y,z,1) position in the world frame (Size: nJoints by 4)
% OUTPUT:
%    centers: Nx3 sphere centers
%    radi:   Nx1 radii for each sphere
function [centers, radi] = stompRobotSphere(X)
nJoints = size(X,1);
center_cell = cell(nJoints,1);
radi_cell = cell(nJoints,1);

% Tuning: sphere radius for link coverage (smaller for iiwa than Kinova)
rad = 0.035;  % meters (suggest 0.03 - 0.04 for KUKA iiwa14)

% Construct the spheres for all links
for k = 1:nJoints
    if k == 1
        parent_joint_position = [0,0,0];
    else
        parent_joint_position = X(k-1, 1:3);
    end
    child_joint_position = X(k, 1:3);

    % number of spheres used to cover the kth link (one every ~rad meters)
    link_len = norm(child_joint_position - parent_joint_position);
    if link_len <= 1e-6
        nSpheres = 1;
    else
        nSpheres = max(1, ceil(link_len / rad) + 1);
    end

    % Calculate the centers of the spheres evenly spaced along the link
    center_cell_k = arrayfun(@(x1, x2) linspace(x1, x2, nSpheres), parent_joint_position', child_joint_position', 'UniformOutput', false);
    center_cell{k} = cell2mat(center_cell_k)';

    % radius of each sphere
    radi_cell{k} = rad * ones(size(center_cell{k},1), 1);
end

centers = cell2mat(center_cell);
radi = cell2mat(radi_cell);
end