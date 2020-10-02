% weight size = 5 x 5 x 1 x4 (float = 8 byte) 
% --> 100 x 8 byte x 8 bits = 6400 bits
S = 100; % s = # of bits required for each edge device
distance_to_device_1 = 100;
distance_to_device_2 = 150;
distance_to_device_3 = 200;
max_distance = 10;
ssq_distance=distance_to_device_1^2+distance_to_device_2^2+distance_to_device_3^2;

trans_power = 1.0;
% previously suggested: 0.3
trans_power_device1 = trans_power*(distance_to_device_1^2)/ssq_distance;
% previously suggested: 0.2
trans_power_device2 = trans_power*(distance_to_device_2^2)/ssq_distance;
% previously suggested: 0.5
trans_power_device3 = trans_power*(distance_to_device_3^2)/ssq_distance;
signal_device1 = rand(1,S) > 0.5;
signal_device2 = rand(1,S) > 0.5;
signal_device3 = rand(1,S) > 0.5;
awgn = sqrt(trans_power_device1)*signal_device1;
awgn = sqrt(trans_power_device2)*signal_device2 + awgn;
awgn = sqrt(trans_power_device3)*signal_device3 + awgn;
awgn_re = 10; 
awgn_device1 = randn(1,S)/awgn_re;
awgn_device2 = randn(1,S)/awgn_re;
awgn_device3 = randn(1,S)/awgn_re;

sys_loss_device1 = 1 + rand/(10*trans_power_device1);
attenna_device1 = trans_power/trans_power_device1 * 1/sys_loss_device1;
attenna_device1 = attenna_device1*(distance_to_device_1^2)/(max_distance^2);
attenna_device1 = 1 - attenna_device1;
sys_loss_device2 = 1 + rand/(10*trans_power_device2);
attenna_device_1 = trans_power/trans_power_device2 * 1/sys_loss_device2;
attenna_device_1 = attenna_device_1*(distance_to_device_2^2)/(max_distance^2);
attenna_device_1 = 1 - attenna_device_1;
sys_loss_device3 = 1 + rand/(10*trans_power_device3);
attenna_device3 = trans_power/trans_power_device3 * 1/sys_loss_device3;
attenna_device3 = attenna_device3*(distance_to_device_3^2)/(max_distance^2);
attenna_device3 = 1 - attenna_device3;

receive_device1 = awgn*attenna_device1 + awgn_device1;
receive_device2 = awgn*attenna_device_1 + awgn_device2;
receive_device3 = awgn*attenna_device3 + awgn_device3;