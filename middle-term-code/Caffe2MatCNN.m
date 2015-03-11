% nonlinear layer binds to the most recent convolutional layer or fully
% connected layer. Loss layer to last fully connected layer
srcPath = 'lung-caffe-model';
desPath = 'lung-caffe.mat';
fileID = fopen(srcPath);
parseDone = 0;
net = [];
l = 1;
while 1
    % layer type
    layer_type = fread(fileID, 1, 'uint32');
    switch layer_type
        case 5  %LayerParameter_LayerType_DATA
            datum = fread(fileID, [1 3], 'uint32');
            net.layers{l} = struct('type', 'i', 'outputmaps', datum(3), 'mapsize', [datum(1) datum(2)]);
            net.layers{l}.type = 'i';
            l = l + 1;
        case 4  %LayerParameter_LayerType_CONVOLUTION
            nblobs = fread(fileID, 1, 'uint32');
            bolb_sz_w = fread(fileID, [1 4], 'uint32');
            
            blob_width = bolb_sz_w(1);
            blob_height = bolb_sz_w(2);
            blob_channels = bolb_sz_w(3);
            blob_num = bolb_sz_w(4);
            assert(blob_channels == net.layers{l-1}.outputmaps, 'wrong architecture');
           
            kernelsize = blob_width;
            kernelArea = blob_width*blob_height;
            kernelVolume = kernelArea*blob_channels;
            net.layers{l}.outputmaps = blob_num;
            
            clear w_1d b;
            w_1d = fread(fileID, prod(bolb_sz_w), 'float');
            %w = reshape(w_1d, bolb_sz_w);
            %w = fread(fileID, bolb_sz_w, 'float');
            bolb_sz_b = fread(fileID, [1 4], 'uint32');
            assert(prod(bolb_sz_b) == blob_num, 'wrong architecture')
            b = fread(fileID, prod(bolb_sz_b), 'float');
            for n = 1 : net.layers{l}.outputmaps  %  output map
                for c = 1 : net.layers{l-1}.outputmaps  %  initials of weights
                    %net.layers{l}.k{i}{j} = w(:, :, j, i)';
                    net.layers{l}.k{n}{c} = zeros(blob_height, blob_width);
                    tmp = zeros(blob_height, blob_width);
                    for kh = 1:blob_height
                        for kw = 1:blob_width
                            %net.layers{l}.k{n}{c}(kh, kw) = w_1d((n-1)*kernelVolume + (kw + (kh-1)*blob_width - 1)*blob_channels + c);
                            tmp(kh, kw) = w_1d((n-1)*kernelVolume + (c-1)*kernelArea + (kh-1)*blob_width + kw);
                        end
                    end
                    for kh = 1:blob_height
                        net.layers{l}.k{n}{c}(kh, :) = tmp(blob_height-kh+1, blob_width:-1:1);
                    end
                end
                net.layers{l}.b{n} = b(n); % bias parameters
            end
            
            
            net.layers{l}.kernelsize = kernelsize;
            mapsize = net.layers{l-1}.mapsize - net.layers{l}.kernelsize + 1;
            net.layers{l}.mapsize = mapsize;
            net.layers{l}.neurons = prod(mapsize)*net.layers{l}.outputmaps;
            net.layers{l}.nonlinear = 0; % linear default
            net.layers{l}.type = 'c';
            l = l + 1;
        case 17 %LayerParameter_LayerType_POOLING only considering max pooling now
             pool_scale = fread(fileID, 1, 'uint32');
             pool_stride = fread(fileID, 1, 'uint32');
             net.layers{l}.scale = pool_scale;
             net.layers{l}.ptype = 'max';
             
             mapsize = net.layers{l-1}.mapsize / pool_scale;
             net.layers{l}.outputmaps = net.layers{l-1}.outputmaps;
             net.layers{l}.mapsize = mapsize;
             net.layers{l}.neurons = prod(mapsize)*net.layers{l}.outputmaps;
             net.layers{l}.type = 's';
             l = l + 1;
        case 18 %LayerParameter_LayerType_RELU
            
            for i = l-1:-1:2
                if net.layers{i}.type == 'c' || net.layers{i}.type == 'f'
                    net.layers{i}.nonlinear = 3; % 3 for relu
                    break;
                end
            end
        case 19 %LayerParameter_LayerType_SIGMOID
            for i = l-1:-1:2
                if net.layers{i}.type == 'c' || net.layers{i}.type == 'f'
                    net.layers{i}.nonlinear = 1; % 1 for sigmoid
                    break;
                end
            end
        case 20 %LayerParameter_LayerType_SOFTMAX
            for i = l-1:-1:2
                if net.layers{i}.type == 'c' || net.layers{i}.type == 'f'
                    net.layers{i}.nonlinear = 4; % 4 for softmax
                    break;
                end
            end
        case 14 %LayerParameter_LayerType_INNER_PRODUCT
            nblobs = fread(fileID, 1, 'uint32');
            bolb_sz_w = fread(fileID, [1 4], 'uint32');
            net.layers{l}.neurons= bolb_sz_w(2);
            clear w_1d b;
            w_1d = fread(fileID, prod(bolb_sz_w), 'float');
            w_1d = w_1d(:);
            w = reshape(w_1d, [bolb_sz_w(1) bolb_sz_w(2)]);
            
            bolb_sz_b = fread(fileID, [1 4], 'uint32');
            b = fread(fileID, prod(bolb_sz_b), 'float');
            assert(prod(bolb_sz_b) == bolb_sz_w(2), 'wrong architecture')
            
            pre_neurons = net.layers{l-1}.neurons;
            if pre_neurons ~= bolb_sz_w(1)
                error('number of neurons should be consistent');
            end
            curr_neurons = net.layers{l}.neurons;
                
            net.layers{l}.b = b(:);
            net.layers{l}.W = w';
            %net.layers{l}.W = reshape(w_1d, [bolb_sz_w(2) bolb_sz_w(1)]);
            net.layers{l}.nonlinear = 0;
            net.layers{l}.type = 'f';
            l = l + 1;
        case 21  %LayerParameter_LayerType_SOFTMAX_LOSS
            net.layers{l-1}.type = 'o';
            net.layers{l-1}.nonlinear = 4; % softmax
            parseDone = 1;
    end
    if 0 ~= parseDone
        break;
    end
end
fclose(fileID);
cnn = net;
save(desPath, 'cnn')
disp('model updated')