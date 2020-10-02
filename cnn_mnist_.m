M = 4;
N = 1;
% M = main(M, N);
% figure2();
figure4();
% PARAM FOR EXPERIMENT
C = [2,3,4,5,6,7,8,9,10];
device1 = device(1);
device2 = device(2);
device3 = device(3);
device1.flop = flop_cnn();  %total number of FLOP at device
device1.c = 2;
device1.f = 1/850;
loc_enerygy = energy_at_device(device1);
loc_time = time_at_device(device1);

function figure2()
    M = [10, 20, 30, 40, 50, 60, 70, 80, 90];
    N = [8, 15, 25];
    gloabal_weight = init_filter();
 
    device1 = device(1);
    device2 = device(2);
    device3 = device(3);
    test = device(0);
    
    aacc1 = [];
    aacc2 = [];
    aacc3 = [];
    for i = 1:length(N)
        n = N(i);
        for j = 1:length(M)
            m = M(j);
            for k=1:m
                [w1, acc1] = local_train(device1.x, device1.y, test.x, test.y, n, gloabal_weight);
                [w2, acc2] = local_train(device2.x, device2.y, test.x, test.y, n, gloabal_weight);
                [w3, acc3] = local_train(device3.x, device3.y, test.x, test.y, n, gloabal_weight);

                gloabal_weight = aggregate_weight(w1, w2, w3);
            end
            aacc1(i, j) = acc1*100;
            aacc2(i, j) = acc2*100;
            aacc3(i, j) = acc3*100;
        end
        disp(aacc1);
        disp(aacc2);
        disp(aacc3);
    end
end
function figure4()
end

function M = main(M, N)

    gloabal_weight = init_filter();
 
%     flop = calculate_flop_cnn();
    device1 = device(1);
    device2 = device(2);
    device3 = device(3);
    test = device(0);
    for i=1:M
        [w1, acc1] = local_train(device1.x, device1.y, test.x, test.y, N, gloabal_weight);
        [w2, acc2] = local_train(device2.x, device2.y, test.x, test.y, N, gloabal_weight);
        [w3, acc3] = local_train(device3.x, device3.y, test.x, test.y, N, gloabal_weight);
        
        gloabal_weight = aggregate_weight(w1, w2, w3);
    end
    print_output(M,acc1, acc2, acc3)
end

%% RANDOM GLOBAL WEIGHT
function weight = init_filter()
%   use 'randn' instead of 'rand':
%   'randn' generates numbers in [-Inf,+Inf], normally distributed (Gaussian);
    weight = randn ([[5 5] 1 4]);
end

%% TRAIN @ LOCAL
function [loc_weight, accuracy] = local_train(X, Y, XTest, YTest, N, global_weight)
    %--- DEFINE LAYER ---%
    layers = [
        imageInputLayer([28 28 1])

        convolution2dLayer(5,4,'Padding',1,'Weights', global_weight)
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(5,12,'Padding',1)
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        fullyConnectedLayer(10)
        softmaxLayer
        classificationLayer];
    
    %--- TRAIN CONFIGURATION ---%
    miniBatchSize = 100;
    Epoch = N;
    options = trainingOptions( 'sgdm',...
        'MaxEpochs',Epoch, ...
        'MiniBatchSize', miniBatchSize,...
        'ValidationData',{XTest,YTest});%, ...
%         'Plots', 'training-progress'); 
%         
%         'InitialLearnRate',0.01,...
%         'Shuffle','every-epoch', ...
%         'ValidationFrequency',30, ...
%         'Verbose',false, ...
%     tic
    net = trainNetwork(X, Y, layers, options);
%     toc
    
%     time = toc;
    loc_weight = net.Layers(2).Weights;
    
    %--- COMPARE WITH TEST DATASET ---%
    predLabelsTest = net.classify(XTest);
    accuracy = sum(predLabelsTest == YTest) / numel(YTest);
end

%% AGGREGATE WEIGHT @SERVER
function global_weight = aggregate_weight(w1, w2, w3)
    global_weight = (w1+w2+w3)/3;
end

% PRINT OUTPUT %
function print_output(M, acc1, acc2, acc3)
    fprintf(':|======================================================================================================================|:\n');
    fprintf(':|  M = %d    |   Train Accuracy                                                                                                        |:\n', M);
    fprintf(':|======================================================================================================================|:\n');
    fprintf(':|  DEVICE 1  |      %.2f %%    |      |:\n', acc1*100);
    fprintf(':|  DEVICE 2  |      %.2f %%    |      |:\n', acc2*100);
    fprintf(':|  DEVICE 3  |      %.2f %%    |      |:\n', acc3*100);
    fprintf(':|  SERVER    |   |   |:\n');
    fprintf(':|======================================================================================================================|:\n');
    fprintf('%.2f\n', acc1*100);
    fprintf('%.2f\n', acc2*100);
    fprintf('%.2f\n', acc3*100);
end

%% Fk %%
function flop = flop(a, dataset)
% ---------------------------------------------------------- %
%| [16]                                                     |%
%| flop = a*|Dk|                                            |%
%|  - a    = # of FLOPs / each data sample per local update |%
%|  - |Dk| = size of data sample                            |%
% ---------------------------------------------------------- %
    flop = a*size(dataset);
end

function flop = flop_cnn()
% ----------------------------------------------------------------------------------------------- %
%| [36]                                                                                          |%
%| CONVOLUTION_FLOPs = (spartial_w x spartial_h) x depth_n x (kernel_w x kernel_h) x depth_(n-1) |%
%| - spartial_w = spartial_w_of_map                                                              |% 
%| - spartial_h = spartial_h_of_map                                                              |% 
%| - depth_n    = current layer depth                                                            |% 
%| - depth_(n-1)= previous layer depth                                                           |% 
%| - kernel_w   = kernel width                                                                   |% 
%| - kernel_h   = kernel height                                                                  |%
%|                                                                                               |%  
%| FULLY-CONNECTED_FLOPs = #output x #input                                                      |%
%| - #output    = # of Fully connected output                                                    |%
%| - #input     = (kernel_w x kernel_h) x depth_(n-1)                                            |% 
% ----------------------------------------------------------------------------------------------- %
       
    convolution_layer1 = (28*28)*4*(5*5)*1;
    convolution_layer2 = (12*12)*12*(5*5)*4;
    fully_connected_layer = (10*1*1)*(4*4*12);
    
    flop = convolution_layer1...
        + convolution_layer2...
        + fully_connected_layer;
end

%% Ek %% = total energy consumption at edge device
function loc_enerygy = energy_at_device(device)
    loc_enerygy = (device.flop/device.c)*device.si*(device.f^2);
end

%% tk %% = computation time duration for each loacal update at edge device
function loc_time = time_at_device(device)
    loc_time = (device.flop/device.c)*(1/device.f);
end

%% T_NOMA = time delay for both local ML-parameter update & uploading 
function time = total_time(M, N)
    local_time = 10;
    up_time = 20
    time = M*(N*local_time+up_time);
end