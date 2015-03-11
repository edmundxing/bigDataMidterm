% run_deep_edge
clear all; clc
addpath('srcDeepScan')
load('muscle-caffe-20.mat', 'cnn');
caffe = 1;
imgPath = '..\data';
files = dir([imgPath '\*.jpg']);
for index = 1:length(files)
    fn = files(index).name;
    img = imread([imgPath '\' fn]);
    imgName = fn(1:end-4);
    
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end
    img = im2double(img);
    
    prob_map_frag = cnn_edge_scanner(cnn, img, caffe);
    imwrite(prob_map_frag, [imgPath '\' imgName '-cnn.png'], 'png');
end