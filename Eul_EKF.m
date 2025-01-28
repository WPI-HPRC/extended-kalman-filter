%% AE5335 - Autonomous Aerial Vehicles
% Attitude Extended Kalman Filter
% Author: Daniel Pearson
% Version: 10/6/2024

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

% Bias Estimation
bG_x = 0;
bG_y = 0;
bG_z = 0;

bA_x = 0;
bA_y = 0;
bA_z = 0;

x_min = [yaw_0; pitch_0; roll_0; bG_x; bG_y; bG_z; bA_x; bA_y; bA_z];
x     = x_min;

%% Initialize Extended Kalman Filter
numStates = length(x_min); % [Yaw, Pitch, Roll]

% Storage
xRecord = nan(numStates, length(imu_data.time));
xRecord(:,1) = x;

PRecord = zeros(numStates, numStates, length(imu_data.time));

%% State Covariance
P = zeros(numStates);

% Initial uncertainity for orientation (Yaw, Pitch, Roll)
P(1:3, 1:3) = eye(3) * imu_data.std_dev.gyr.^2;

% Initial uncertainty for gyroscope bias
P(4:6, 4:6) = eye(3) * 0.1;

% Initial uncertainty for accelerometer bias
P(7:9, 7:9) = eye(3) * 0.1;

PRecord(:,:,1) = P;

%% Process Noise Covariance
std_dev_gyrBias  = 0.001;
std_dev_accBias  = 0.001;

Q_k = zeros(numStates);

% Q_k(1:3, 1:3) = eye(3) * imu_data.std_dev.gyr.^2;
Q_k(1:3, 1:3) = eye(3) * 0.0002;

Q_k(4:6, 4:6) = eye(3) * std_dev_gyrBias.^2;

Q_k(7:9, 7:9) = eye(3) * std_dev_accBias.^2;

for i = 2:length(imu_data.time)
    dt = imu_data.time(i) - imu_data.time(i-1);
    t = t + dt;

    % Input vector
    u_k1   = imu_data.gyroscope(:,i-1);
    u_k    = imu_data.gyroscope(:,i);
    u_k12  = 0.5*(u_k1 + u_k);

    % Apply RK4 for state prediction
    k1 = dt*measurementFunction(x, u_k1);
    k2 = dt*measurementFunction(x + 0.5*k1, u_k12);
    k3 = dt*measurementFunction(x + 0.5*k2, u_k12);
    k4 = dt*measurementFunction(x + k3, u_k);

    x_min = x + ((1/6)*k1 + (1/3)*k2 + (1/3)*k3 + (1/6)*k4);

    % Compute Jacobian of the measurement function
    F = measurementJacobian(x, u_k);

    % State transition matrix (discrete form)
    phi = (eye(numStates) + F * dt);

    % Process noise covariance update
    P_min = phi*P*phi' + Q_k;

    %% Update Step

    % Gravity Correction
    h_grav = [g*sin(x(2)); 
             -g*sin(x(3))*cos(x(2)); 
             -g*cos(x(3))*cos(x(2))] + [x(7); x(8); x(9)];
 
    H_grav = [
        0, g*cos(x(2)), 0,                                0, 0, 0, 1, 0, 0;
        0, g*sin(x(2))*sin(x(3)), -g*cos(x(2))*cos(x(3)), 0, 0, 0, 0, 1, 0;
        0, g*cos(x(3))*sin(x(2)), g*cos(x(2))*sin(x(3)), 0, 0, 0, 0, 0, 1;
    ];

    R_grav = eye(3) * imu_data.std_dev.acc.^2;

    H = [
        H_grav;
    ];

    % R = blkdiag(R_comp, R_grav);
    R = R_grav;

    S = H*P_min*H' + R;

    K = P_min * H' / S;

    % Apply compass bias
    z_accel = imu_data.accelerometer(:,i) - [x(7); x(8); x(9)];

    z = z_accel;

    h = h_grav;

    x = x_min + K * (z - h);
    P = (eye(numStates) - K*H) * P_min;

    % Store the state and covariance matrix at each step
    xRecord(:,i) = x;
    PRecord(:, :, i) = P;
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
title('EKF Yaw Covariance');
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
title('EKF Pitch Covariance');
grid on;
pitch_error = truth_data.euler_angles.pitch - xRecord(2,:);
plot(imu_data.time, pitch_error, 'r--');  % Add pitch error in dashed red line
legend('\sigma_1', '-\sigma_1', 'Pitch Error');
hold off;

% Roll Covariance
subplot(3,1,3);  % 3 rows, 1 column, 3rd plot
plot(imu_data.time, P_roll, imu_data.time, -P_roll, 'LineWidth', 2);
hold on;
xlabel('Time (s)');
ylabel('Error (deg)');
title('EKF Roll Covariance');
grid on;
roll_error = truth_data.euler_angles.roll - xRecord(3,:);
plot(imu_data.time, roll_error, 'r--');  % Add roll error in dashed red line
legend('\sigma_1', '-\sigma_1', 'Roll Error');
hold off;

figsize = [0, 0.04, 0.8, 0.8];
figure('Units', 'Normalized', 'InnerPosition', figsize, 'OuterPosition', figsize, 'Name', 'EKF Estimate');

subplot(311); plot(imu_data.time, truth_data.euler_angles.yaw*180/pi, 'LineWidth', 2); hold on;
% subplot(311); plot(imu_data.time, x_hat_store(1,:)*180/pi, 'LineWidth', 2); 
subplot(311); plot(imu_data.time, xRecord(1,:)*180/pi, '--', 'LineWidth', 2); 
% subplot(311); plot(imu_data.time, z_fiction(1,:)*180/pi, 'LineWidth', 0.5); 
xlabel('Time (s)'); ylabel('\psi (deg)'); grid on;
set(gca, 'FontSize', 20, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
legend('True', 'EKF estimate')

subplot(312); plot(imu_data.time, truth_data.euler_angles.pitch*180/pi, 'LineWidth', 2); hold on;
% subplot(312); plot(imu_data.time, x_hat_store(2,:)*180/pi, 'LineWidth', 2);
subplot(312); plot(imu_data.time, xRecord(2,:)*180/pi, '--', 'LineWidth', 2); 
% subplot(312); plot(imu_data.time, z_fiction(2,:)*180/pi, 'LineWidth', 0.5); 
xlabel('Time (s)'); ylabel('\theta (deg)');  grid on;
set(gca, 'FontSize', 20, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
legend('True', 'EKF estimate')

subplot(313); plot(imu_data.time, truth_data.euler_angles.roll*180/pi, 'LineWidth', 2); hold on;
% subplot(313); plot(timestamps, x_hat_store(3,:)*180/pi, 'LineWidth', 2);
subplot(313); plot(imu_data.time, xRecord(3,:)*180/pi, '--', 'LineWidth', 2); 
% subplot(313); plot(timestamps, z_fiction(3,:)*180/pi, 'LineWidth', 0.5); 
xlabel('Time (s)'); ylabel('\phi (deg)');  grid on;
set(gca, 'FontSize', 20, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
legend('True', 'EKF estimate')

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

%% Measurement Jacobian
function F = measurementJacobian(x, u)
    p = u(1) - x(4);
    q = u(2) - x(5);
    r = u(3) - x(6);

    F = [ 0,  (cos(x(3))*sin(x(2))*(r - x(6)))/cos(x(2))^2 + (sin(x(2))*sin(x(3))*(q - x(5)))/cos(x(2))^2,  (cos(x(3))*(q - x(5)))/cos(x(2)) - (sin(x(3))*(r - x(6)))/cos(x(2)),  0,  -sin(x(3))/cos(x(2)),  -cos(x(3))/cos(x(2)),  0,  0,  0;
      0,  0,  -cos(x(3))*(r - x(6)) - sin(x(3))*(q - x(5)),  0,  -cos(x(3)),  sin(x(3)),  0,  0,  0;
      0,  cos(x(3))*(r - x(6)) + sin(x(3))*(q - x(5)) + (cos(x(3))*sin(x(2))^2*(r - x(6)))/cos(x(2))^2 + (sin(x(2))^2*sin(x(3))*(q - x(5)))/cos(x(2))^2,  (cos(x(3))*sin(x(2))*(q - x(5)))/cos(x(2)) - (sin(x(2))*sin(x(3))*(r - x(6)))/cos(x(2)),  -1,  -(sin(x(2))*sin(x(3)))/cos(x(2)),  -(cos(x(3))*sin(x(2)))/cos(x(2)),  0,  0,  0;
      0,  0,  0,  0,  0,  0,  0,  0,  0;
      0,  0,  0,  0,  0,  0,  0,  0,  0;
      0,  0,  0,  0,  0,  0,  0,  0,  0;
      0,  0,  0,  0,  0,  0,  0,  0,  0;
      0,  0,  0,  0,  0,  0,  0,  0,  0;
      0,  0,  0,  0,  0,  0,  0,  0,  0 ];

end