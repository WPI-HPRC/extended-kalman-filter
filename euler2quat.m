function q_total = euler2quat(roll, pitch, yaw)
    % Function to convert Roll, Pitch, Yaw angles to a quaternion
    % Inputs: roll, pitch, yaw (in radians)
    q_total = [
        cos(0.5 * yaw) * cos(0.5 * pitch) * cos(0.5 * roll) + sin(0.5 * yaw) * sin(0.5 * pitch) * sin(0.5 * roll);
        cos(0.5 * yaw) * cos(0.5 * pitch) * sin(0.5 * roll) - sin(0.5 * yaw) * sin(0.5 * pitch) * cos(0.5 * roll);
        cos(0.5 * yaw) * sin(0.5 * pitch) * cos(0.5 * roll) + sin(0.5 * yaw) * cos(0.5 * pitch) * sin(0.5 * roll);
        sin(0.5 * yaw) * cos(0.5 * pitch) * cos(0.5 * roll) - cos(0.5 * yaw) * sin(0.5 * pitch) * sin(0.5 * roll);
    ];

end