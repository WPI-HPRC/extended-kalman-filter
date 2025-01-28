%% AE5335 - Autonomous Aerial Vehicles
% Trajectory Fake
% Author: Daniel Pearson
% Version: 10/10/2024

clear variables; close all; clc;

%% Constants
g = 9.80665; % [m/s^2] Gravitational Acceleration

mag_ned = [50; 0; -20]; % [uT] NED magnetic field vector

%% Time Keeping
t = linspace(0, 500, 10000); % [s] Time Span

%% Yaw Generation
yaw_noise = 20 * randn(1, length(t));   % Random noise for yaw with a large variance
yaw_smooth = smoothdata(yaw_noise, 'gaussian', 1500);   % Smooth over a window of 300 points
shape_factor = 1 + 0.2 * sin(2 * pi * 0.001 * t);  % Varying shape factor
yaw = shape_factor .* yaw_smooth;   % Apply the shape factor to the yaw trajectory

%% Pitch Generation
pitch_noise = 20 * randn(1, length(t));   % Random noise for yaw with a large variance
pitch_smooth = smoothdata(pitch_noise, 'gaussian', 1500);   % Smooth over a window of 300 points
shape_factor = 1 + 0.2 * sin(2 * pi * 0.001 * t);  % Varying shape factor
pitch = shape_factor .* pitch_smooth;   % Apply the shape factor to the yaw trajectory

%% Roll Generation
roll_noise = 20 * randn(1, length(t));   % Random noise for yaw with a large variance
roll_smooth = smoothdata(roll_noise, 'gaussian', 1500);   % Smooth over a window of 300 points
shape_factor = 1 + 0.2 * sin(2 * pi * 0.001 * t);  % Varying shape factor
roll = shape_factor .* roll_smooth;   % Apply the shape factor to the yaw trajectory

%% Fit cubic splines to the yaw, pitch, and roll data to smooth the trajectory further
% Downsample to control points and then apply spline fitting
control_points_time = linspace(0, 500, 50); % Fit 50 control points alone spline
yaw_control_points = interp1(t, yaw, control_points_time, 'pchip');
pitch_control_points = interp1(t, pitch, control_points_time, 'pchip');
roll_control_points = interp1(t, roll, control_points_time, 'pchip');

% Spline interpolation for yaw, pitch, and roll
yaw = spline(control_points_time, yaw_control_points, t);
pitch = spline(control_points_time, pitch_control_points, t);
roll = spline(control_points_time, roll_control_points, t);

% Initialize angular velocity vectors (wx, wy, wz)
wx = zeros(1, length(t));
wy = zeros(1, length(t));
wz = zeros(1, length(t));

% Initialize accelerometer vector
ax = zeros(1, length(t));
ay = zeros(1, length(t));
az = zeros(1, length(t));

% Initialize magnetometer readings (mx, my, mz)
mx = zeros(1, length(t));
my = zeros(1, length(t));
mz = zeros(1, length(t));

% Bias values for each axis (can simulate sensor drift)
bias_wx = 0.0175;  % Bias for wx (rad/s)
bias_wy = -0.02;   % Bias for wy (rad/s)
bias_wz = 0.015;   % Bias for wz (rad/s)

bias_ax = 0.049;   % Bias for ax (m/s/s)
bias_ay = -0.07;   % Bias for ay (m/s/s)
bias_az = 0.024;   % Bias for az (m/s/s)

bias_mx = 0.3;     % Bias for mx (uT);
bias_my = -0.4;     % Bias for mY (uT);
bias_mz = 0.2;     % Bias for mZ (uT);
% bias_wx = 0;  % Bias for wx (rad/s)
% bias_wy = 0;   % Bias for wy (rad/s)
% bias_wz = 0;   % Bias for wz (rad/s)
% 
% bias_ax = 0;   % Bias for ax (m/s/s)
% bias_ay = 0;   % Bias for ay (m/s/s)
% bias_az = 0;   % Bias for az (m/s/s)
% 
% bias_mx = 0;     % Bias for mx (uT);
% bias_my = 0;     % Bias for mY (uT);
% bias_mz = 0;     % Bias for mZ (uT);

% Noise standard deviation (to simulate random measurement noise)
std_dev_gyro = 0.0005;  % Gyroscope noise (rad/s)
std_dev_accel = 0.0049;  % Accelerometer noise (m/s/s)
std_dev_mag = 0.05;      % Magnetometer noise (μT)

% Compute initial accelerometer readings based on the initial angles
ax(1) = g * sin(pitch(1));
ay(1) = -g*sin(roll(1))*cos(pitch(1));
az(1) = -g*cos(roll(1))*cos(pitch(1));

% Compute angular velocity from Euler rates (derivatives of Euler angles)
for i = 2:length(t)
    dt = t(i) - t(i-1);
    
    % Derivatives of Euler angles
    d_yaw = (yaw(i) - yaw(i-1)) / dt;
    d_pitch = (pitch(i) - pitch(i-1)) / dt;
    d_roll = (roll(i) - roll(i-1)) / dt;
    
    % ZYX Euler Angular Velocity (ωx, ωy, ωz)
    wx(i) = d_roll - sin(pitch(i)) * d_yaw;
    wy(i) = cos(roll(i)) * d_pitch + sin(roll(i)) * cos(pitch(i)) * d_yaw;
    wz(i) = -sin(roll(i)) * d_pitch + cos(roll(i)) * cos(pitch(i)) * d_yaw;
    
    % Add bias and noise to each angular velocity
    wx(i) = wx(i) + bias_wx + std_dev_gyro * randn;
    wy(i) = wy(i) + bias_wy + std_dev_gyro * randn;
    wz(i) = wz(i) + bias_wz + std_dev_gyro * randn;

    % Accelerometer readings
    ax(i) = g * sin(pitch(i));
    ay(i) = -g*sin(roll(i))*cos(pitch(i));
    az(i) = -g*cos(roll(i))*cos(pitch(i));

    ax(i) = ax(i) + bias_ax + std_dev_accel * randn;
    ay(i) = ay(i) + bias_ay + std_dev_accel * randn;
    az(i) = az(i) + bias_az + std_dev_accel * randn;

    % Magnetometer Readings
    R = angle2dcm(yaw(i), pitch(i), roll(i), 'ZYX');
    mag_body = R * mag_ned;

    % Magnetometer readings with bias and noise
    mx(i) = mag_body(1) + bias_mx + std_dev_mag * randn;
    my(i) = mag_body(2) + bias_my + std_dev_mag * randn;
    mz(i) = mag_body(3) + bias_mz + std_dev_mag * randn;

end

imu_data.time = t;
imu_data.accelerometer = [ax; ay; az];
imu_data.gyroscope     = [wx; wy; wz];

imu_data.std_dev.gyr = std_dev_gyro;
imu_data.std_dev.acc = std_dev_accel;

truth_data.euler_angles.yaw = yaw;
truth_data.euler_angles.pitch = pitch;
truth_data.euler_angles.roll = roll;

% Plot Euler angles
figure;
subplot(3, 1, 1); plot(imu_data.time, truth_data.euler_angles.yaw); title('Yaw (ψ)'); xlabel('Time (s)'); ylabel('Angle (rad)');
subplot(3, 1, 2); plot(imu_data.time, truth_data.euler_angles.pitch); title('Pitch (θ)'); xlabel('Time (s)'); ylabel('Angle (rad)');
subplot(3, 1, 3); plot(imu_data.time, truth_data.euler_angles.roll); title('Roll (φ)'); xlabel('Time (s)'); ylabel('Angle (rad)');

% Plot Angular Velocities (IMU readings with bias and noise)
figure;
subplot(3, 1, 1); plot(imu_data.time, imu_data.gyroscope(1,:)); title('Angular Velocity wx (rad/s) - with Bias and Noise'); xlabel('Time (s)'); ylabel('wx (rad/s)');
subplot(3, 1, 2); plot(imu_data.time, imu_data.gyroscope(2,:)); title('Angular Velocity wy (rad/s) - with Bias and Noise'); xlabel('Time (s)'); ylabel('wy (rad/s)');
subplot(3, 1, 3); plot(imu_data.time, imu_data.gyroscope(3,:)); title('Angular Velocity wz (rad/s) - with Bias and Noise'); xlabel('Time (s)'); ylabel('wz (rad/s)');

% Plot Accelerometer Readings
figure;
subplot(3,1,1); plot(imu_data.time, imu_data.accelerometer(1,:)); title('Accelerometer X (m/s^2) - with Bias and Noise'); xlabel('Time (s)'); ylabel('ax (m/s^2)');
subplot(3,1,2); plot(imu_data.time, imu_data.accelerometer(1,:)); title('Accelerometer Y (m/s^2) - with Bias and Noise'); xlabel('Time (s)'); ylabel('ay (m/s^2)');
subplot(3,1,3); plot(imu_data.time, imu_data.accelerometer(1,:)); title('Accelerometer Z (m/s^2) - with Bias and Noise'); xlabel('Time (s)'); ylabel('az (m/s^2)');

save('FakeTraj.mat', 'imu_data', 'truth_data');