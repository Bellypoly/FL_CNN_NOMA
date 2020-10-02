M = 1;
N = 8;
init(M, N);

% DEVICE 1 CONFIGURATION
F1 = calculate_flop_cnn();  %total number of FLOP at device

% DEVICE 2 CONFIGURATION
F2 = calculate_flop_cnn();  %total number of FLOP at device

% DEVICE 3 CONFIGURATION
F3 = calculate_flop_cnn();  %total number of FLOP at device

function init(M, N)

    gloabal_weight = init_filter();
    [X_device1,Y_device1, X_device2, Y_device2, X_device3, Y_device3, XTest, YTest] = prepare_data();
    
%     flop = calculate_flop_cnn();

    for i=1:M
        [w1, acc1] = local_train(X_device1, Y_device1, XTest, YTest, N, gloabal_weight);
        [w2, acc2] = local_train(X_device2, Y_device2, XTest, YTest, N, gloabal_weight);
        [w3, acc3] = local_train(X_device3, Y_device3, XTest, YTest, N, gloabal_weight);
        
        gloabal_weight = aggregate_weight(w1, w2, w3);
    end
    print_output(M,acc1, acc2, acc3)
end