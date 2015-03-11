% img
clear all; clc
addpath('srcDeepScan')
img = imread('..\data\Bmal1 KO#3-2.jpg');
%%
maxImgDim = 1000;
if length(img) > maxImgDim
    img = imresize(img, maxImgDim/length(img));
end
if size(img,3) == 1
    img = cat(3, img, img, img);
end
% if size(img, 3) == 3
%     img = rgb2gray(img);
% end
%img = double(img);
img = im2double(img);

%load('edge_scanner_51.mat', 'cnn');
load('muscle-caffe-20.mat', 'cnn');
%load('muscle-big.mat', 'cnn');
%load('caffe_cnn.mat', 'cnn');
size(img)
timeID = tic;
caffe = 1;
prob_map_frag = cnn_edge_scanner(cnn, img, caffe);
toc(timeID)

% patch_size = 31;
% half_patch_size = floor(patch_size/2);
% H = size(img, 1);
% W = size(img, 2);
% 
% imgExpand = padarray(img, [half_patch_size half_patch_size], 'symmetric');
% timeID = tic;
% [scores, timeStat] = cnnff_image_pred(cnn, imgExpand, [patch_size patch_size]);
% prob_map_frag = scores;
%size(prob_map_frag) - size(img)

% prob_map_frag = zeros(H, W);
% 
% 
% for irow = half_patch_size+1:H-half_patch_size
%     for jcol = half_patch_size+1:W-half_patch_size
%         prob_map_frag(irow, jcol) = scores(irow-half_patch_size, jcol-half_patch_size);
%     end
% end
frag_time = toc(timeID)
figure, imshow(prob_map_frag)
%%
% ucm = contours2ucm(prob_map_frag);
% figure, imshow(ucm)

