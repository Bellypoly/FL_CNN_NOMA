% TEST 60k | TRAIN 10k
%% LOAD MNIST DATA %%
ImagesTrain = 'dataset\train-images.idx3-ubyte';
LabelsTrain = 'dataset\train-labels.idx1-ubyte';
ImagesTest = 'dataset\t10k-images.idx3-ubyte';
LabelsTest = 'dataset\t10k-labels.idx1-ubyte';

%% READ BINARY FILE and CREATE IMAGE ARRAY
XTrain = processImagesMNIST(ImagesTrain);
YTrain = processLabelsMNIST(LabelsTrain);
XTest = processImagesMNIST(ImagesTest);
YTest = processLabelsMNIST(LabelsTest);

% X_device1 = XTest(:,:,1,1:300);
% Y_device1 = YTest(1:300);
% X_device2 = XTest(:,:,1,301:600);
% Y_device2 = YTest(301:600);
% X_device3 = XTest(:,:,1,601:1000);
% Y_device3 = YTest(601:1000);
% % check by whos 'var_name'

%% DEFINE LAYER %%
layers = [
    imageInputLayer([28 28 1])
	
    convolution2dLayer(5,4,'Padding',1)
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
miniBatchSize = 8192;
Epoch = 4;
options = trainingOptions( 'sgdm',...
    'MaxEpochs',Epoch, ...
    'MiniBatchSize', miniBatchSize,...
    'ValidationData',{XTest,YTest},...
    'Plots', 'training-progress');
% options = trainingOptions('sgdm',...
%     'InitialLearnRate',0.01,...
%     'MaxEpochs',8, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',imdsValidation, ...
%     'ValidationFrequency',30, ...
%     'Verbose',false, ...
%     'Plots','training-progress');

net = trainNetwork(XTrain, YTrain, layers, options);

predLabelsTest = net.classify(XTest);
accuracy = sum(predLabelsTest == YTest) / numel(YTest);

%%
% digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', 'nndatasets','DigitDataset');
% imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders',true,'LabelSource','foldernames');

% figure;
% perm = randperm(10000,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imds.Files{perm(i)});
% end

% labelCount = countEachLabel(imds)
% 
% img = readimage(imds,1);
% size(img)
% 
% numTrainFiles = 750;
% [imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
% 
% %% DEFINE LAYER 1 %%
% layers = [
%     imageInputLayer([28 28 1])
%     
%     convolution2dLayer(3,8,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];
% 
% %% DEFINE LAYER 2 %%
% layers = [
%     imageInputLayer([28 28 1])
% 	
%     convolution2dLayer(3,16,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
% 	
%     maxPooling2dLayer(2,'Stride',2)
% 	
%     convolution2dLayer(3,32,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
% 	
%     maxPooling2dLayer(2,'Stride',2)
% 	
%     convolution2dLayer(3,64,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
% 	
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];
% %% TRAIN CONFIGURATION %%
% options = trainingOptions('sgdm',...
%     'InitialLearnRate',0.01,...
%     'MaxEpochs',25, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',imdsValidation, ...
%     'ValidationFrequency',30, ...
%     'Verbose',false, ...
%     'Plots','training-progress');
% 
% net = trainNetwork(imdsTrain,layers,options);
% 
% YPred = classify(net,imdsValidation);
% YValidation = imdsValidation.Labels;
% 
% accuracy = sum(YPred == YValidation)/numel(YValidation)