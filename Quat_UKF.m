%% AE5335 - Autonomous Aerial Vehicles
% Attitude Unscented Kalman Filter with Quaternions
% Author: Daniel Pearson
% Version: 10/10/2024

% clear variables; close all; clc;
clear variables; clc;

%% Constants
g = 9.80665; % [m/s^2]

%% Data Loading
load('FakeTraj_ICM42688.mat');

t = 0; % [s] Start Time
dt = imu_data.time(2) - imu_data.time(1);

% Initialize state using accelerometer (assuming upright)
acX_0 = imu_data.accelerometer(1,1);
acY_0 = imu_data.accelerometer(2,1);
acZ_0 = imu_data.accelerometer(3,1);

% Calculate the initial quaternion from accelerometer data
pitch_0 = asin(acX_0 / g);
roll_0  = atan(acY_0 / acZ_0);
yaw_0 = truth_data.euler_angles.yaw(1);  % Initialize yaw from truth data

q_0 = cos(0.5 * yaw_0) * cos(0.5 * pitch_0) * cos(0.5 * roll_0) + sin(0.5 * yaw_0) * sin(0.5 * pitch_0) * sin(0.5 * roll_0);
q_1 = cos(0.5 * yaw_0) * cos(0.5 * pitch_0) * sin(0.5 * roll_0) - sin(0.5 * yaw_0) * sin(0.5 * pitch_0) * cos(0.5 * roll_0);
q_2 = cos(0.5 * yaw_0) * sin(0.5 * pitch_0) * cos(0.5 * roll_0) + sin(0.5 * yaw_0) * cos(0.5 * pitch_0) * sin(0.5 * roll_0);
q_3 = sin(0.5 * yaw_0) * cos(0.5 * pitch_0) * cos(0.5 * roll_0) - cos(0.5 * yaw_0) * sin(0.5 * pitch_0) * sin(0.5 * roll_0);

% Bias Initialization
bG_x = 0;
bG_y = 0;
bG_z = 0;

bA_x = 0;
bA_y = 0;
bA_z = 0;

x = [q_0; q_1; q_2; q_3; bG_x; bG_y; bG_z; bA_x; bA_y; bA_z];
numStates = length(x);

%% Initialize UKF Parameters
alpha = 1e-3;   % Spread of sigma points
beta  = 2;      % For Gaussian distributions
kappa = 0;      % Secondary scaling parameter
lambda = alpha^2 * (numStates + kappa) - numStates;

% Initialize Weights for mean and covariance
W_m = [lambda / (numStates + lambda), repmat(1 / (2 * (numStates + lambda)), 1, 2 * numStates)];
W_c = W_m;
W_c(1) = W_m(1) + (1 - alpha^2 + beta);

% Storage
xRecord = nan(numStates, length(imu_data.time));
xRecord(:,1) = x;

PRecord = zeros(numStates, numStates, length(imu_data.time));

%% State Covariance
P = eye(numStates) * 0.1;  % Initial state covariance

% Gyroscope and accelerometer measurement noise
R_grav = eye(3) * imu_data.std_dev.acc.^2;

%% Process Noise Covariance
std_dev_gyrBias  = 0.001;
std_dev_accBias  = 0.001;

Q_k = zeros(numStates);
Q_k(1:4, 1:4) = eye(4) * 0.0002;      % Process noise for quaternion
Q_k(5:7, 5:7) = eye(3) * std_dev_gyrBias.^2;  % Gyro bias process noise
Q_k(8:10, 8:10) = eye(3) * std_dev_accBias.^2; % Accelerometer bias process noise

PRecord(:,:,1) = P;

%% UKF Iteration Loop
for i = 2:length(imu_data.time)
    dt = imu_data.time(i) - imu_data.time(i-1);
    t = t + dt;

    % Step 1: Generate sigma points
    [X_sigma, P_sigma] = unscentedSigmaPoints(x, P, lambda, numStates);

    % Step 2: Propagate sigma points through process model
    X_sigma_pred = processModel(X_sigma, imu_data.gyroscope(:,i), dt);

    % Step 3: Predict mean and covariance
    [x_min, P_min] = unscentedMeanCov(X_sigma_pred, W_m, W_c, Q_k);

    %% Measurement Update Step
    % Compute sigma points for measurement
    Z_sigma = measurementModel(X_sigma_pred);

    % Predict measurement mean and covariance
    [z_pred, P_zz] = unscentedMeanCov(Z_sigma, W_m, W_c, R_grav);

    % Cross covariance
    P_xz = unscentedCrossCov(X_sigma_pred, x_min, Z_sigma, z_pred, W_c);

    % Kalman gain
    K = P_xz / P_zz;

    % Measurement residual
    z_accel = imu_data.accelerometer(:,i) - [x(8); x(9); x(10)];
    z = z_accel;

    % Update state and covariance
    x = x_min + K * (z - z_pred);
    P = P_min - K * P_zz * K';

    % x = x_min;
    % 
    % P = P_min;

    % Normalize quaternion
    x(1:4) = x(1:4) / norm(x(1:4));

    % Store state and covariance
    xRecord(:,i) = x;
    PRecord(:,:,i) = P;
end

%% Covariance Plotting with Truth Data

% Pre-allocate for quaternion truth data
q_truth = zeros(4, length(imu_data.time));

% Loop over each time step and convert Euler angles to quaternions
for i = 1:length(imu_data.time)
    roll = truth_data.euler_angles.roll(i);
    pitch = truth_data.euler_angles.pitch(i);
    yaw = truth_data.euler_angles.yaw(i);
    
    % Convert truth Euler angles to quaternion using your euler2quat function
    q_truth(:, i) = euler2quat(roll, pitch, yaw);
end

% Compute quaternion estimation errors
q_err = xRecord(1:4,:) - q_truth;

% Extract diagonal elements of the covariance matrix for quaternions
P_q1 = squeeze(PRecord(1,1,:));
P_q2 = squeeze(PRecord(2,2,:));
P_q3 = squeeze(PRecord(3,3,:));

% Plot quaternion q0 error and covariance
figure('Name','Quaternion Covariance and Errors with Truth Data');

subplot(3,1,1);
plot(imu_data.time, P_q1, 'b--', imu_data.time, -P_q1, 'b--', 'LineWidth', 1.5);
hold on;
plot(imu_data.time, q_err(1,:), 'r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Error q_0');
title('UKF Quaternion q_0 Error and Covariance');
legend('Covariance Bound', 'Error');
grid on;
hold off;

% Plot quaternion q1 error and covariance
subplot(3,1,2);
plot(imu_data.time, P_q2, 'b--', imu_data.time, -P_q2, 'b--', 'LineWidth', 1.5);
hold on;
plot(imu_data.time, q_err(2,:), 'r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Error q_1');
title('UKF Quaternion q_1 Error and Covariance');
legend('Covariance Bound', 'Error');
grid on;
hold off;

% Plot quaternion q2 error and covariance
subplot(3,1,3);
plot(imu_data.time, P_q3, 'b--', imu_data.time, -P_q3, 'b--', 'LineWidth', 1.5);
hold on;
plot(imu_data.time, q_err(3,:), 'r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Error q_2');
title('UKF Quaternion q_2 Error and Covariance');
legend('Covariance Bound', 'Error');
grid on;
hold off;


%% UKF Function Definitions
function [X_sigma, P_sigma] = unscentedSigmaPoints(x, P, lambda, numStates)
    % Generate sigma points
    P_sigma = chol((numStates + lambda) * P)';
    X_sigma = [x, bsxfun(@plus, x, P_sigma), bsxfun(@minus, x, P_sigma)];
end

function X_sigma_pred = processModel(X_sigma, u, dt)
    % Propagate each sigma point through quaternion kinematics
    numPoints = size(X_sigma, 2);
    X_sigma_pred = zeros(size(X_sigma));
    for k = 1:numPoints
        q = X_sigma(1:4,k);  % Extract quaternion
        omega = u - X_sigma(5:7,k);  % Gyroscope measurement minus bias

        % Quaternion kinematics
        omega_quat = [0; omega];
        q_dot = 0.5 * quatmultiply(q', omega_quat')';

        % Update quaternion and normalize
        q_new = q + dt * q_dot;
        q_new = q_new / norm(q_new);

        X_sigma_pred(:,k) = [q_new; X_sigma(5:10,k)];  % Update biases and state
    end
end

function [x_mean, P_mean] = unscentedMeanCov(X_sigma, W_m, W_c, Q)
    % Compute weighted mean and covariance for quaternions
    x_mean = X_sigma * W_m';
    P_mean = Q;
    for k = 1:size(X_sigma, 2)
        P_mean = P_mean + W_c(k) * (X_sigma(:,k) - x_mean) * (X_sigma(:,k) - x_mean)';
    end
end

function P_xz = unscentedCrossCov(X_sigma_pred, x_min, Z_sigma, z_pred, W_c)
    % Compute cross covariance
    P_xz = zeros(size(X_sigma_pred,1), size(Z_sigma,1));
    for k = 1:size(X_sigma_pred, 2)
        P_xz = P_xz + W_c(k) * (X_sigma_pred(:,k) - x_min) * (Z_sigma(:,k) - z_pred)';
    end
end

function Z_sigma = measurementModel(X_sigma)
    % Transform gravity from world to body frame using quaternions
    g = 9.80665;
    numPoints = size(X_sigma, 2);
    Z_sigma = zeros(3, numPoints);
    for k = 1:numPoints
        q = X_sigma(1:4,k);
        gravity_world = [0; 0; g];
        gravity_body = quatrotate(q', gravity_world')' + [X_sigma(8, k); X_sigma(9, k); X_sigma(10, k)];
        Z_sigma(:,k) = gravity_body + X_sigma(8:10,k);  % Accelerometer bias added
    end
end
