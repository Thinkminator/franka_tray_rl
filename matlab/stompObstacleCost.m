function cost = stompObstacleCost(sphere_centers, radius, voxel_world, vel)
% Computes obstacle cost for a set of spheres using voxel world signed EDT.
% Inputs:
%  - sphere_centers: Nx3
%  - radius: Nx1 or 1x1
%  - voxel_world: struct with fields voxel_size, Env_size, world_size, sEDT
%  - vel: Nx1 speeds for each sphere
%
% Output:
%  - cost: scalar

safety_margin = 0.05; % meters
cost = 0;

voxel_world_sEDT = voxel_world.sEDT;
world_size = voxel_world.world_size; % [nx ny nz]

% env corner (metric origin of the voxel grid)
env_corner = voxel_world.Env_size(1,:); % [xmin, ymin, zmin]
env_corner_vec = repmat(env_corner, size(sphere_centers,1), 1);

% voxel size may be vector
vsize = voxel_world.voxel_size;
if isscalar(vsize)
    vsize = [vsize, vsize, vsize];
end

% compute voxel indices (ceil as before)
idx = ceil((sphere_centers - env_corner_vec) ./ vsize);

% clamp indices to valid ranges to avoid indexing errors
idx(:,1) = min(max(idx(:,1), 1), world_size(1));
idx(:,2) = min(max(idx(:,2), 1), world_size(2));
idx(:,3) = min(max(idx(:,3), 1), world_size(3));

% Now compute distances using clamped indices
N = size(idx,1);
d = zeros(N,1);
for i = 1:N
    try
        d(i) = voxel_world_sEDT(idx(i,1), idx(i,2), idx(i,3));
    catch
        % If still fails, set a large negative distance (inside obstacle)
        d(i) = -min(world_size); 
    end
end

% Ensure radius is column vector
if isscalar(radius)
    radius = radius * ones(N,1);
else
    radius = radius(:);
    if numel(radius) ~= N
        radius = repmat(radius(1), N, 1); % fallback
    end
end

% compute per-sphere cost according to Eq (13) style: only positive intrusion counts
cost_array = max(safety_margin + radius - d, 0) .* abs(vel);
cost = sum(cost_array);

end