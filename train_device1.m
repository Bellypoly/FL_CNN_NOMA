function t1 = train_device1(M, N)
%     M = 1;
%     N = 25;
    gloabal_weight = init_filter();
    [X_device1,Y_device1, XTest, YTest] = prepare_data();

    for i=1:M
        [w1, t1] = local_train(X_device1, Y_device1, XTest, YTest, N, gloabal_weight);
    end
    options = simset('SrcWorkspace','current');
    sim('FL_CNN_NOMA',[],options)
end

function [X_device,Y_device, XTest, YTest] = prepare_data()
    %% LOAD MNIST DATA %%
    Images = 'dataset\t10k-images.idx3-ubyte';
    Labels = 'dataset\t10k-labels.idx1-ubyte';

    %% READ BINARY FILE and CREATE IMAGE ARRAY
    XX = processImagesMNIST(Images);
    YY = processLabelsMNIST(Labels);

    X_device = XX(:,:,1,1:300);        % DATASET MNIST@DEVICE 1 train : 0-300 | 1000 for test
    Y_device = YY(1:300);
%     X_device2 = XX(:,:,1,301:600);      % DATASET MNIST@DEVICE 2 train : 301-600 | 1000 for test
%     Y_device2 = YY(301:600);
%     X_device3 = XX(:,:,1,601:1000);     % DATASET MNIST@DEVICE 3 train : 601-1000 | 1000 for test
%     Y_device3 = YY(601:1000);
    XTest = XX(:,:,1,1001:2000);        % DATASET MNIST train : 1001-2000 | 1000 for test
    YTest = YY(1001:2000);
end

%% GLOBAL WEIGHT
function weight = init_filter()
    weight = 0.0001 * randn([[5 5] 1 4]);
%     disp(weight)
end

%% TRAIN @ LOCAL
function [loc_weight, time] = local_train(X, Y, XTest, YTest, N, global_weight)
    %% DEFINE LAYER %%
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

        convolution2dLayer(4,12,'Padding',1)
        batchNormalizationLayer
        reluLayer

        fullyConnectedLayer(10)
        softmaxLayer
        classificationLayer];
    
    %% TRAIN CONFIGURATION %%
    miniBatchSize = 100;
    Epoch = N;
    options = trainingOptions( 'sgdm',...
        'MaxEpochs',Epoch, ...
        'MiniBatchSize', miniBatchSize,...
        'ValidationData',{XTest,YTest}, ...
        'Plots', 'training-progress');
    %     'InitialLearnRate',0.01,...
    %     'Shuffle','every-epoch', ...
    %     'ValidationFrequency',30, ...
    %     'Verbose',false, ...
    tic
    net = trainNetwork(X, Y, layers, options);
    toc
    
    time = toc;
    loc_weight = net.Layers(2).Weights
    
    %% COMPARE WITH TEST DATASET
    predLabelsTest = net.classify(XTest);
    accuracy = sum(predLabelsTest == YTest) / numel(YTest);
end

%% AGGREGATE WEIGHT @SERVER
function global_weight = aggregate_weight(w1, w2, w3)
    global_weight = (w1+w2+w3)/3;
end
