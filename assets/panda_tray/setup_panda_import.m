urdfFile = 'panda_tray_cylinder_fixed.urdf'; % Make sure this path is correct
robot = importrobot(urdfFile);
robot.DataFormat = 'column'; % Set data format for joint positions/velocities

% Display robot details (optional)
showdetails(robot);

% Visualize the robot (optional)
figure;
show(robot);
view(3); % 3D view
axis equal;
light;