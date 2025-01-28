%% AE5335 - Autonomous Aerial Vehicles
% Extended Kalman Filter
% Author: Daniel Pearson
% Version: 10/9/2024

clear variables; close all; clc;

%% Constants
g = 9.80665; % [m/s^2]

%% Load IMU Data
load('FakeTraj.mat');

t = 0;
dt = imu_data.time(2) - imu_data.time(1);


% Initialize state using accelerometer
acX_0 = imu_data.accelerometer(1,1);
acY_0 = imu_data.accelerometer(2,1);
acZ_0 = imu_data.accelerometer(3,1);

% pitch_0 = asin(acX_0 / g);
% roll_0  = asin(acY_0 / acZ_0);
pitch_0 = truth_data.euler_angles.pitch(1);
roll_0  = truth_data.euler_angles.roll(1);
yaw_0   = truth_data.euler_angles.yaw(1);

% Initialize as true initial quaternion
% q_0 = euler2quat(pitch_0, roll_0, yaw_0)';
q_0 = eul2quat([yaw_0; pitch_0; roll_0]')';

quaternions_true = eul2quat([truth_data.euler_angles.yaw; truth_data.euler_angles.pitch; truth_data.euler_angles.roll]', 'ZYX')';

% Bias Estimation
bG_x = 0;
bG_y = 0;
bG_z = 0;

bA_x = 0;
bA_y = 0;
bA_z = 0;

w_ib_x = 0;
w_ib_y = 0;
w_ib_z = 0;

x_min = [q_0; w_ib_x; w_ib_y; w_ib_z; bG_x; bG_y; bG_z; bA_x; bA_y; bA_z];
x     = x_min;

%% Initialize Extended Kalman Filter
numStates = length(x_min);

% Storage
xRecord = nan(numStates, length(imu_data.time));
xRecord(:,1) = x;

PRecord = zeros(numStates, numStates, length(imu_data.time));

%% State Covariance
P = zeros(numStates);

% Initial uncertainity for orientation
P(1:4, 1:4) = eye(4) * imu_data.std_dev.gyr.^2;

% Initial uncertainity for angular rate
P(5:7, 5:7) = eye(3) * imu_data.std_dev.gyr.^2;

% Initial uncertainty for gyroscope bias
P(8:10, 8:10) = eye(3) * 0.01;

% Initial uncertainty for accelerometer bias
P(11:13, 11:13) = eye(3) * 0.01;

PRecord(:,:,1) = P;


%% Process Noise Covariance
std_dev_gyrBias  = 0.001;
std_dev_accBias  = 0.001;

Q_k = zeros(numStates);

Q_k(1:4, 1:4) = eye(4) * imu_data.std_dev.gyr.^2;

Q_k(5:7, 5:7) = eye(3) * imu_data.std_dev.gyr.^2;

Q_k(8:10, 8:10) = eye(3) * std_dev_gyrBias.^2;

Q_k(11:13, 11:13) = eye(3) * std_dev_accBias.^2;


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

    x_min(1:4) = x_min(1:4) / norm(x_min(1:4));

    % Compute Jacobian of the measurement function
    F = measurementJacobian(x, u_k);

    % State transition matrix (discrete form)
    phi = (eye(numStates) + F * dt);

    % Process noise covariance update
    P_min = phi*P*phi' + Q_k;

    %% Update Step

    G_NED = [0; 0; -g]; 

    q = x_min(1:4);

    R_TB = quat2rot(q');

    h_accel = R_TB' * G_NED + [x(8); x(9); x(10)];

    z_accel = imu_data.accelerometer(1:3, i) - [x(8); x(9); x(10)];

    H_accel = [
        2*g*x(3), -2*g*x(4),  2*g*x(1), -2*g*x(2), 0, 0, 0, 0, 0, 0, 1, 0, 0;
       -2*g*x(2), -2*g*x(1), -2*g*x(4), -2*g*x(3), 0, 0, 0, 0, 0, 0, 0, 1, 0;
       -4*g*x(1),  0,         0,        -4*g*x(4), 0, 0, 0, 0, 0, 0, 0, 0, 1;
    ];

    % H_accel = [ 
    %     2*g*x_min(3), -2*g*x_min(4),  2*g*x_min(1), -2*g*x_min(2), 0, 0, 0, 1, 0, 0
    %    -2*g*x_min(2), -2*g*x_min(1), -2*g*x_min(4), -2*g*x_min(3), 0, 0, 0, 0, 1, 0
    %    -4*g*x_min(1),             0,             0, -4*g*x_min(4), 0, 0, 0, 0, 0, 1];

    R_accel = eye(3) * imu_data.std_dev.acc;

    z = z_accel;
    h = h_accel;
    H = H_accel;
    R = R_accel;

    S = H*P_min*H' + R;
    K = P_min * H' / S;

    x = x_min + K * (z-h);

    P = (eye(numStates) - K*H) * P_min;

    x = x_min;
    P = P_min;
    

    % z = z_accel;

    % h = h_grav;

    % x = x_min + K * (z - h);
    % P = (eye(numStates) - K*H) * P_min;

    % x = x_min;
    % % 
    % % P = P_min;

    % Store the state and covariance matrix at each step
    xRecord(:,i) = x;
    PRecord(:, :, i) = P;

    
end

% Extract P_min(1,1,i) for plotting
% Extract P_min(1,1,i) for plotting
P_qw = sqrt(squeeze(PRecord(1,1,:)));
P_qx = sqrt(squeeze(PRecord(2,2,:)));
P_qy = sqrt(squeeze(PRecord(3,3,:)));
P_qz = sqrt(squeeze(PRecord(4,4,:)));

P_gbx = sqrt(squeeze(PRecord(5,5,:)));
P_gby = sqrt(squeeze(PRecord(6,6,:)));
P_gbz = sqrt(squeeze(PRecord(7,7,:)));


%% Covariance Plotting - Quaternion
figure('Name','Quaternion Covariance and Errors');

% Quaternion q_w Covariance
subplot(4,1,1);  % 4 rows, 1 column, 1st plot
plot(imu_data.time, P_qw, imu_data.time, -P_qw, 'LineWidth', 2);
hold on;
xlabel('Time (s)');
ylabel('P(q_w,q_w)');
title('q_w Covariance');
grid on;
qw_error = quaternions_true(1,:) - xRecord(1,:);
plot(imu_data.time, qw_error, 'r--');  % Add q_w error in dashed red line
legend('\sigma_{q_w}', '-\sigma_{q_w}', 'q_w Error');
hold off;

% Quaternion q_x Covariance
subplot(4,1,2);  % 4 rows, 1 column, 2nd plot
plot(imu_data.time, P_qx, imu_data.time, -P_qx, 'LineWidth', 2);
hold on;
xlabel('Time (s)');
ylabel('P(q_x,q_x)');
title('q_x Covariance');
grid on;
qx_error = quaternions_true(2,:) - xRecord(2,:);
plot(imu_data.time, qx_error, 'r--');  % Add q_x error in dashed red line
legend('\sigma_{q_x}', '-\sigma_{q_x}', 'q_x Error');
hold off;

% Quaternion q_y Covariance
subplot(4,1,3);  % 4 rows, 1 column, 3rd plot
plot(imu_data.time, P_qy, imu_data.time, -P_qy, 'LineWidth', 2);
hold on;
xlabel('Time (s)');
ylabel('P(q_y,q_y)');
title('q_y Covariance');
grid on;
qy_error = quaternions_true(3,:) - xRecord(3,:);
plot(imu_data.time, qy_error, 'r--');  % Add q_y error in dashed red line
legend('\sigma_{q_y}', '-\sigma_{q_y}', 'q_y Error');
hold off;

% Quaternion q_z Covariance
subplot(4,1,4);  % 4 rows, 1 column, 4th plot
plot(imu_data.time, P_qz, imu_data.time, -P_qz, 'LineWidth', 2);
hold on;
xlabel('Time (s)');
ylabel('P(q_z,q_z)');
title('q_z Covariance');
grid on;
qz_error = quaternions_true(4,:) - xRecord(4,:);
plot(imu_data.time, qz_error, 'r--');  % Add q_z error in dashed red line
legend('\sigma_{q_z}', '-\sigma_{q_z}', 'q_z Error');
hold off;

%% Covariance Plotting - Gyro Bias
% figure('Name','Gyro Bias Covariance and Errors');
% 
% % Quaternion q_w Covariance
% subplot(4,1,1);  % 4 rows, 1 column, 1st plot
% plot(imu_data.time, P_gbx, imu_data.time, -P_gbx, 'LineWidth', 2);
% hold on;
% xlabel('Time (s)');
% ylabel('P(q_w,q_w)');
% title('q_w Covariance');
% grid on;
% % qw_error = quaternions_true(1,:) - xRecord(1,:);
% gbx_error = imu_data.time * imu_data.bias.gyroX - xRecord(5,:);
% plot(imu_data.time, qw_error, 'r--');  % Add q_w error in dashed red line
% legend('\sigma_{q_w}', '-\sigma_{q_w}', 'q_w Error');
% hold off;
% 
% % Quaternion q_x Covariance
% subplot(4,1,2);  % 4 rows, 1 column, 2nd plot
% plot(imu_data.time, P_qx, imu_data.time, -P_qx, 'LineWidth', 2);
% hold on;
% xlabel('Time (s)');
% ylabel('P(q_x,q_x)');
% title('q_x Covariance');
% grid on;
% qx_error = quaternions_true(2,:) - xRecord(2,:);
% plot(imu_data.time, qx_error, 'r--');  % Add q_x error in dashed red line
% legend('\sigma_{q_x}', '-\sigma_{q_x}', 'q_x Error');
% hold off;
% 
% % Quaternion q_y Covariance
% subplot(4,1,3);  % 4 rows, 1 column, 3rd plot
% plot(imu_data.time, P_qy, imu_data.time, -P_qy, 'LineWidth', 2);
% hold on;
% xlabel('Time (s)');
% ylabel('P(q_y,q_y)');
% title('q_y Covariance');
% grid on;
% qy_error = quaternions_true(3,:) - xRecord(3,:);
% plot(imu_data.time, qy_error, 'r--');  % Add q_y error in dashed red line
% legend('\sigma_{q_y}', '-\sigma_{q_y}', 'q_y Error');
% hold off;

%% EKF Quaternion Estimate Plotting
figsize = [0, 0.04, 0.8, 0.8];
figure('Units', 'Normalized', 'InnerPosition', figsize, 'OuterPosition', figsize, 'Name', 'EKF Quaternion Estimate');

% Plot q_w (scalar part)
subplot(411); plot(imu_data.time, quaternions_true(1,:), 'LineWidth', 2); hold on;
subplot(411); plot(imu_data.time, xRecord(1,:), '--', 'LineWidth', 2); 
xlabel('Time (s)'); ylabel('q_w'); grid on;
set(gca, 'FontSize', 20, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
legend('True', 'EKF estimate')

% Plot q_x (x component of vector part)
subplot(412); plot(imu_data.time, quaternions_true(2,:), 'LineWidth', 2); hold on;
subplot(412); plot(imu_data.time, xRecord(2,:), '--', 'LineWidth', 2); 
xlabel('Time (s)'); ylabel('q_x');  grid on;
set(gca, 'FontSize', 20, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
legend('True', 'EKF estimate')

% Plot q_y (y component of vector part)
subplot(413); plot(imu_data.time, quaternions_true(3,:), 'LineWidth', 2); hold on;
subplot(413); plot(imu_data.time, xRecord(3,:), '--', 'LineWidth', 2); 
xlabel('Time (s)'); ylabel('q_y');  grid on;
set(gca, 'FontSize', 20, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
legend('True', 'EKF estimate')

% Plot q_z (z component of vector part)
subplot(414); plot(imu_data.time, quaternions_true(4,:), 'LineWidth', 2); hold on;
subplot(414); plot(imu_data.time, xRecord(4,:), '--', 'LineWidth', 2); 
xlabel('Time (s)'); ylabel('q_z');  grid on;
set(gca, 'FontSize', 20, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
legend('True', 'EKF estimate')


function f = measurementFunction(x, u)

    p = u(1) - x(8); q = u(2) - x(9); r = u(3) - x(10);

    f_q = 0.5 * [
        -x(2) -x(3) -x(4);
         x(1) -x(4)  x(3);
         x(4)  x(1) -x(2);
        -x(3)  x(2)  x(1);
    ] * x(5:7);

    f = [f_q; p; q; r; 0; 0; 0; 0; 0; 0];

end

function F = measurementJacobian(x, u)
    p = u(1) - x(8);
    q = u(2) - x(9);
    r = u(3) - x(10);

    gbx = x(8);
    gby = x(9);
    gbz = x(10);

    qw = x(1);
    qx = x(2);
    qy = x(3);
    qz = x(4);

    w_ib_x = x(5);
    w_ib_y = x(6);
    w_ib_z = x(7);

    F = [
        0, -w_ib_x/2, -w_ib_y/2, -w_ib_z, -qx/2, -qy/2, -qz/2, 0, 0, 0, 0, 0, 0;
        w_ib_x/2, 0, w_ib_z/2, -w_ib_y/2, qw/2, -qz/2, qy/2, 0, 0, 0, 0, 0, 0;
        w_ib_y/2, -w_ib_z/2, 0, -w_ib_x/2, qz/2, qw/2, -qx/2, 0, 0, 0, 0, 0, 0;
        w_ib_z/2, w_ib_y/2, -w_ib_x/2, 0, -qy/2, qx/2, qw/2, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    ];

    % F = [
    %               0, gbx/2 - p/2, gby/2 - q/2, gbz/2 - r/2,  qx/2,  qy/2,  qz/2, 0, 0, 0;
    %     p/2 - gbx/2,           0, r/2 - gbz/2, gby/2 - q/2, -qw/2,  qz/2, -qy/2, 0, 0, 0;
    %     q/2 - gby/2, gbz/2 - r/2,           0, p/2 - gbx/2, -qz/2, -qw/2,  qx/2, 0, 0, 0;
    %     r/2 - gbz/2, q/2 - gby/2, gbx/2 - p/2,           0,  qy/2, -qx/2, -qw/2, 0, 0, 0;
    %               0,           0,           0,           0,     0,     0,     0, 0, 0, 0;
    %               0,           0,           0,           0,     0,     0,     0, 0, 0, 0;
    %               0,           0,           0,           0,     0,     0,     0, 0, 0, 0;
    %               0,           0,           0,           0,     0,     0,     0, 0, 0, 0;
    %               0,           0,           0,           0,     0,     0,     0, 0, 0, 0;
    %               0,           0,           0,           0,     0,     0,     0, 0, 0, 0;
    % ];
end