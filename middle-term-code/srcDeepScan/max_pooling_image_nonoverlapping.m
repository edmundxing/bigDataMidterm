function [frags_out offset_out] = max_pooling_image_nonoverlapping(frags, net, l, offset_pre)
    
    k = net.layers{l}.scale;
    k2 = k*k;
    frags_out = cell(1, k2*length(frags)); % k2 times frags
    offset_out = cell(1, k2*length(frags)); % each frags has one offset
  
    offset = zeros(k2, 2);
    for i = 1:k
        for j = 1:k
            offset(k*(i-1) + j, :) = [i-1 j-1];
        end
    end
    
    
    for i_frag = 1:length(frags)
        maps_in = frags{i_frag};
        [nrow, ncol, num_maps] = size(maps_in);
        
        for i = 1:k2
            start_x = 1 + offset(i, 1);
            start_y = 1 + offset(i, 2);

            nrow_new = floor((nrow - start_y + 1)/k);
            ncol_new = floor((ncol - start_x + 1)/k);
    
            maps_out = zeros(nrow_new, ncol_new, num_maps);
            for j = 1:num_maps
                maps_out(:,:,j) = max_pooling_nonoverlapping(maps_in(:,:,j), k, start_x, start_y, nrow_new, ncol_new);
            end
            
            
            frag_id = k2*(i_frag-1) + i;
            frags_out{frag_id} = maps_out;
            
            if isempty(offset_pre)
                offset_out{frag_id} = [offset(i, :) k];
            else
                offset_out{frag_id} = [offset_pre{i_frag}; offset(i, :) k];
            end
            
        end
    end
end