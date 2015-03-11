function [scores, imgFeats] = cnnff_image_pred(net, x, patch_size, caffe, featLayers)
    if nargin < 5
        featLayers = [];
    end
    if nargin < 4
        caffe = 0;
    end
    % input x is image in any sizes and the patch used in training
    % get patch size
    img_size = size(x(:,:,1));
    patch_num = img_size - patch_size + 1;
    
    n = numel(net.layers);
    offsets = [];
    map_size = patch_size;
    
    frags = cell(1, 1);
    frags{1} = x;
    rsp = [];
    
    extract_feat = 0;
    imgFeats = [];
    featCount = 1;
    for l = 2 : n   %  for each layer
        %net.layers{l}.type
        switch net.layers{l}.type
            case 'c' % c layer will not change frags
                map_size = map_size - net.layers{l}.kernelsize + 1; % the feature map size in trained net
                frags_out = cell(1, length(frags)); % for convolutional layer, #frags will not change
                inputmaps = net.layers{l - 1}.outputmaps;
                outputmaps = net.layers{l}.outputmaps;
                for i_frag = 1:length(frags_out)
                    curr_in = frags{i_frag}; % current frag, all feature maps
                    img_map_size = size(curr_in(:,:,1)) - net.layers{l}.kernelsize + 1;
                    curr_out = zeros(img_map_size(1), img_map_size(2), outputmaps);
                    for j = 1 : outputmaps    %  for each output map
                        %  create temp output map
                        z = zeros(img_map_size);

                        for i = 1 : inputmaps   %  for each input map
                            %  convolve with corresponding kernel and add to temp output map
                            z = z + conv2(curr_in(:,:,i), net.layers{l}.k{j}{i}, 'valid');
                        end
                        %  add bias, pass through nonlinearity
                        
                        curr_out(:,:,j) = nonlinear_unit(z + net.layers{l}.b{j}, net.layers{l}.nonlinear);
                    end
                    
                    frags_out{i_frag} = curr_out;
                end
                frags = frags_out;
                
                if any(l == featLayers)
                    feat_layer = gather_frags(frags, offsets, img_size, patch_size, map_size);
                    imgFeats{featCount}.feat = feat_layer;
                    imgFeats{featCount}.len = size(feat_layer, 1);
                    imgFeats{featCount}.layerNum = l;
                    featCount = featCount + 1;
                end
                
            case 's' % this is the key step for image based prediction
                
                if isfield(net.layers{l}, 'mapsize')
                    map_size = net.layers{l}.mapsize;
                else
                    map_size = floor(map_size ./ net.layers{l}.scale);
                end
                
                
                if strcmp(net.layers{l}.ptype, 'max')
                    % max pooling
                        [frags, offsets] = max_pooling_image(frags, net, l, offsets);
                else % avg
                        [frags, offsets] = max_pooling_image(frags, net, l, offsets);
                end
                if any(l == featLayers)
                    feat_layer = gather_frags(frags, offsets, img_size, patch_size, map_size);
                    imgFeats{featCount}.feat = feat_layer;
                    imgFeats{featCount}.len = size(feat_layer, 1);
                    imgFeats{featCount}.layerNum = l;
                    featCount = featCount + 1;
                end
            case 'f'
                if net.layers{l-1}.type ~= 'f' % previous layer is also fully connected layer
                    %tGatherStart = tic;
                    
                    rsp = gather_frags(frags, offsets, img_size, patch_size, map_size, caffe);
                    %tGatherUsed = toc(tGatherStart)
                end
                net.layers{l}.nonlinear
                rsp = nonlinear_unit(net.layers{l}.W * rsp + repmat(net.layers{l}.b, 1, size(rsp, 2)), net.layers{l}.nonlinear);
                
                if any(l == featLayers)
                    feat_layer = rsp;
                    imgFeats{featCount}.feat = feat_layer;
                    imgFeats{featCount}.len = size(feat_layer, 1);
                    imgFeats{featCount}.layerNum = l;
                    featCount = featCount + 1;
                end
                    
            case 'o'
                if net.layers{l-1}.type ~= 'f' % previous layer is also fully connected layer
                    rsp = gather_frags(frags, offsets, img_size, patch_size, [map_size, map_size]);
                end
                
                rsp = nonlinear_unit(net.layers{l}.W * rsp + repmat(net.layers{l}.b, 1, size(rsp, 2)), net.layers{l}.nonlinear);
                size(rsp)
                if caffe == 1
                    scores = rsp(2, :) ./ sum(rsp, 1);
                else
                    scores = rsp(1, :) ./ sum(rsp, 1);
                end
                
                scores = reshape(scores, [patch_num(2) patch_num(1)]);
                scores = scores';
        end
        
    end
    if extract_feat == 1
        imgFeats(1) = [];
    end
    
end
