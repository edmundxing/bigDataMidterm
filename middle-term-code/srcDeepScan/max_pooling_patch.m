function map_out = max_pooling_patch(map_in, k, stride, start_x, start_y, nrow_new, ncol_new)
  
    map_out = zeros(nrow_new, ncol_new);
    for i_new = 1:nrow_new
        for j_new = 1:ncol_new
            max_val = -Inf;
            old_start_y = start_y + stride*(i_new-1);
            old_start_x = start_x + stride*(j_new-1);
            
            for i = 1:k
                for j = 1:k
                    if map_in(old_start_y+i-1, old_start_x+j-1) > max_val
                        max_val = map_in(old_start_y+i-1, old_start_x+j-1);
                    end
                end
            end
            map_out(i_new, j_new) = max_val;
        end
    end
end