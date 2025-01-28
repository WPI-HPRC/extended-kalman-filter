%% AE5335 - Autonomous Aerial Vehicles
% Attitude Unscented Kalman Filter
% Author: Daniel Pearson
% Version: 10/10/2024

% clear variables; close all; clc;
clear variables; clc;
% close all;

%% Constants
g = 9.80665; % [m/s^2]

%% Data Loading
load('FakeTraj.mat');

t = 0; % [s] Start Time
dt = imu_data.time(2) - imu_data.time(1);

% Initialize state using accelerometer
acX_0 = imu_data.accelerometer(1,1);
acY_0 = imu_data.accelerometer(2,1);
acZ_0 = imu_data.accelerometer(3,1);

pitch_0 = asin(acX_0 / g);
roll_0  = atan(acY_0 / acZ_0);

% Calculate the tilt-compensated heading
yaw_0 = truth_data.euler_angles.yaw(1);

% Bias Initialization
bG_x = 0;
bG_y = 0;
bG_z = 0;

bA_x = 0;
bA_y = 0;
bA_z = 0;

x = [yaw_0; pitch_0; roll_0; bG_x; bG_y; bG_z; bA_x; bA_y; bA_z];
numStates = length(x);

%% Initialize UKF Parameters
alpha = 1e-3;   % Spread of sigma points
% UKF Scaling Parameters
beta  = 5;
kappa = 0.2;
lambda = alpha^2 * (numStates + kappa) - numStates;

% Initialize Weights for mean and covariance
W_m = [lambda / (numStates + lambda), repmat(1 / (2 * (numStates + lambda)), 1, 2 * numStates)];
W_c = W_m;
W_c(1) = W_m(1) + (1 - alpha^2 + beta);
    
% Storage
xRecord = nan(numStates, length(imu_data.time));
xRecord(:,1) = x;

PRecord = zeros(numStates, numStates, length(imu_data.time));

R_grav = eye(3) * imu_data.std_dev.acc.^2;

%% State Covariance
P = zeros(numStates);
P(1:3, 1:3) = eye(3) * imu_data.std_dev.gyr.^2;
P(4:6, 4:6) = eye(3) * 0.1;
P(7:9, 7:9) = eye(3) * 0.1;

PRecord(:,:,1) = P;

%% Process Noise Covariance
std_dev_gyrBias  = 0.001;
std_dev_accBias  = 0.001;

Q_k = zeros(numStates);
Q_k(1:3, 1:3) = eye(3) * 0.0002;
Q_k(4:6, 4:6) = eye(3) * std_dev_gyrBias.^2;
Q_k(7:9, 7:9) = eye(3) * std_dev_accBias.^2;

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
    z_accel = imu_data.accelerometer(:,i) - [x(7); x(8); x(9)];
    z = z_accel;

    % Update state and covariance
    x = x_min + K * (z - z_pred);
    P = P_min - K * P_zz * K';

    % Store state and covariance
    xRecord(:,i) = x;
    PRecord(:,:,i) = P;
end

% Extract P_min(1,1,i) for plotting
P_yaw = sqrt(squeeze(PRecord(1,1,:)));
P_pitch = sqrt(squeeze(PRecord(2,2,:)));
P_roll = sqrt(squeeze(PRecord(3,3,:)));
P_biasX = sqrt(squeeze(PRecord(4,4,:)));

%% Covariance Plotting
figure('Name','Covariances and Errors');

% Yaw Covariance
subplot(3,1,1);  % 3 rows, 1 column, 1st plot
plot(imu_data.time, P_yaw, imu_data.time, -P_yaw, 'LineWidth', 2);
hold on;
xlabel('Time (s)');
ylabel('Error (deg)');
title('UKF Yaw Covariance');
grid on;
yaw_error = truth_data.euler_angles.yaw - xRecord(1,:);
plot(imu_data.time, yaw_error, 'r--');  % Add yaw error in dashed red line
legend('\sigma_1', '-\sigma_1', '\epsilon_\psi');
hold off;

% Pitch Covariance
subplot(3,1,2);  % 3 rows, 1 column, 2nd plot
plot(imu_data.time, P_pitch, imu_data.time, -P_pitch, 'LineWidth', 2);
hold on;
xlabel('Time (s)');
ylabel('Error (deg)');
title('UKF Pitch Covariance');
grid on;
pitch_error = truth_data.euler_angles.pitch - xRecord(2,:);
plot(imu_data.time, pitch_error, 'r--');  % Add pitch error in dashed red line
legend('P(2,2)', '-P(2,2)', 'Pitch Error');
hold off;

% Roll Covariance
subplot(3,1,3);  % 3 rows, 1 column, 3rd plot
plot(imu_data.time, P_roll, imu_data.time, -P_roll, 'LineWidth', 2);
hold on;
xlabel('Time (s)');
ylabel('Error (deg)');
title('UKF Roll Covariance');
grid on;
roll_error = truth_data.euler_angles.roll - xRecord(3,:);
plot(imu_data.time, roll_error, 'r--');  % Add roll error in dashed red line
legend('P(3,3)', '-P(3,3)', 'Roll Error');
hold off;

figsize = [0, 0.04, 0.8, 0.8];
figure('Units', 'Normalized', 'InnerPosition', figsize, 'OuterPosition', figsize, 'Name', 'UKF Estimate');

subplot(311); plot(imu_data.time, truth_data.euler_angles.yaw*180/pi, 'LineWidth', 2); hold on;
% subplot(311); plot(imu_data.time, x_hat_store(1,:)*180/pi, 'LineWidth', 2); 
subplot(311); plot(imu_data.time, xRecord(1,:)*180/pi, '--', 'LineWidth', 2); 
% subplot(311); plot(imu_data.time, z_fiction(1,:)*180/pi, 'LineWidth', 0.5); 
xlabel('Time (s)'); ylabel('\psi (deg)'); grid on;
set(gca, 'FontSize', 20, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
legend('True', 'UKF Estimate')

subplot(312); plot(imu_data.time, truth_data.euler_angles.pitch*180/pi, 'LineWidth', 2); hold on;
% subplot(312); plot(imu_data.time, x_hat_store(2,:)*180/pi, 'LineWidth', 2);
subplot(312); plot(imu_data.time, xRecord(2,:)*180/pi, '--', 'LineWidth', 2); 
% subplot(312); plot(imu_data.time, z_fiction(2,:)*180/pi, 'LineWidth', 0.5); 
xlabel('Time (s)'); ylabel('\theta (deg)');  grid on;
set(gca, 'FontSize', 20, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
legend('True', 'UKF Estimate')

subplot(313); plot(imu_data.time, truth_data.euler_angles.roll*180/pi, 'LineWidth', 2); hold on;
% subplot(313); plot(timestamps, x_hat_store(3,:)*180/pi, 'LineWidth', 2);
subplot(313); plot(imu_data.time, xRecord(3,:)*180/pi, '--', 'LineWidth', 2); 
% subplot(313); plot(timestamps, z_fiction(3,:)*180/pi, 'LineWidth', 0.5); 
xlabel('Time (s)'); ylabel('\phi (deg)');  grid on;
set(gca, 'FontSize', 20, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
legend('True', 'UKF Estimate')


%% UKF Function Definitions
function [X_sigma, P_sigma] = unscentedSigmaPoints(x, P, lambda, numStates)
    % Calculate sigma points
    P_sigma = chol((numStates + lambda) * P)';
    X_sigma = [x, bsxfun(@plus, x, P_sigma), bsxfun(@minus, x, P_sigma)];
end

function X_sigma_pred = processModel(X_sigma, u, dt)
    % Propagate each sigma point through process model
    numPoints = size(X_sigma, 2);
    X_sigma_pred = zeros(size(X_sigma));
    for k = 1:numPoints
        X_sigma_pred(:,k) = X_sigma(:,k) + dt * measurementFunction(X_sigma(:,k), u);
    end
end

function [x_mean, P_mean] = unscentedMeanCov(X_sigma, W_m, W_c, Q)
    % Compute weighted mean and covariance
    x_mean = X_sigma * W_m';
    P_mean = Q;
    for k = 1:size(X_sigma, 2)
        P_mean = P_mean + W_c(k) * (X_sigma(:,k) - x_mean) * (X_sigma(:,k) - x_mean)';
    end
end

function Z_sigma = measurementModel(X_sigma)

    g = 9.80665; % [m/s/s]
    
    % Compute measurement model for each sigma point
    numPoints = size(X_sigma, 2);
    Z_sigma = zeros(3, numPoints);
    for k = 1:numPoints
        Z_sigma(:,k) = [g * sin(X_sigma(2,k));
                        -g * sin(X_sigma(3,k)) * cos(X_sigma(2,k));
                        -g * cos(X_sigma(3,k)) * cos(X_sigma(2,k))] + [X_sigma(7,k); X_sigma(8,k); X_sigma(9, k)];
    end
end

function P_xz = unscentedCrossCov(X_sigma, x_mean, Z_sigma, z_mean, W_c)
    % Compute cross covariance between state and measurement
    P_xz = zeros(size(X_sigma, 1), size(Z_sigma, 1));
    for k = 1:size(X_sigma, 2)
        P_xz = P_xz + W_c(k) * (X_sigma(:,k) - x_mean) * (Z_sigma(:,k) - z_mean)';
    end
end

%% Measurement Function
function f = measurementFunction(x,u)
    p = u(1) - x(4);
    q = u(2) - x(5);
    r = u(3) - x(6);

    yaw_dot = [0, sin(x(3))/cos(x(2)), cos(x(3))/cos(x(2))];
    pit_dot = [0, (cos(x(3))*cos(x(2)))/cos(x(2)), (-sin(x(3))*cos(x(2)))/cos(x(2))];
    rol_dot = [cos(x(2))/cos(x(2)), (sin(x(3))*sin(x(2)))/cos(x(2)), (cos(x(3))*sin(x(2)))/cos(x(2))];

    eul_dot = [yaw_dot; pit_dot; rol_dot] * [p; q; r];
    
    f = [eul_dot; 0; 0; 0; 0; 0; 0;];
end