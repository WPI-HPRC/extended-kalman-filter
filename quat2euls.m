function [roll, pitch, yaw] = quat2euls(q)
    % Function to convert a quaternion to Roll, Pitch, Yaw angles
    % Input: q (quaternion in the format [qw, qx, qy, qz])
    % Outputs: roll, pitch, yaw (in radians)

    qw = q(1);
    qx = q(2);
    qy = q(3);
    qz = q(4);

    % Roll (X-axis rotation)
    roll = atan2(2*(qw*qx + qy*qz), 1 - 2*(qx^2 + qy^2));
    
    % Pitch (Y-axis rotation)
    pitch = asin(2*(qw*qy - qz*qx));
    
    % Yaw (Z-axis rotation)
    yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2));
end
