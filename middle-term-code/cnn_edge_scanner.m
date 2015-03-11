function [cnnEdge, deepFeat] = cnn_edge_scanner(cnn, img, caffe, featLayers)
    if nargin < 4
        featLayers = [];
    end
    if nargin < 3
        caffe = 0;
    end
    patch_size = cnn.layers{1}.mapsize;
    patch_size = patch_size(1);
    
    half_patch_size = floor(patch_size/2);
    imgExpand = padarray(img, [half_patch_size half_patch_size], 'symmetric');
    [cnnEdge, imgFeats] = cnnff_image_pred(cnn, imgExpand, [patch_size patch_size], caffe, featLayers);
     deepFeat = [];
     for i = 1:length(imgFeats)
        deepFeat = [deepFeat; imgFeats{i}.feat];
     end
    
end