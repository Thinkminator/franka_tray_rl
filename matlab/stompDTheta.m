% dtheta: estimated gradient
% em: 1 by nJoints cell, each cell is nSamples by nDiscretize matrix
function dtheta = stompDTheta(trajProb, em)

nJoints = length(em);
nDiscretize = size(trajProb, 2);
% variable declaration
dtheta = zeros(nJoints, nDiscretize);

%% TODO: iterate over all joints to compute dtheta: (complete your code according to the STOMP algorithm) 
for m = 1:nJoints
    Em = em{m};  % nSamples x nDiscretize noise for joint m
    for t = 1:nDiscretize
        w = trajProb(:,t);              % nSamples x 1 weights
        dtheta(m,t) = sum(w .* Em(:,t)) / (sum(w) + eps);
    end


    % Keep endpoints fixed (zero gradient)
    dtheta(:,1) = 0;
    dtheta(:,end) = 0;
end