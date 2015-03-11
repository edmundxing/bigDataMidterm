function frags_reunion = gather_frags(frags, offsets, img_size, patch_size, map_size, caffe)
    if nargin < 6
        caffe = 0;
    end
    % cmopute number of patches
    patch_num = img_size - patch_size + 1;
    N = size(frags{1}, 3);
    if length(map_size) < 2
        map_size = [map_size map_size];
    end
    frags_reunion = zeros(N*prod(map_size), prod(patch_num));
    
    for i_frag = 1:length(frags)
        offset = offsets{i_frag}; % one offset for each fragment
        img_maps = frags{i_frag}; % 
        [nrows, ncols, ~] = size(img_maps);
        
        nrows = nrows - map_size(1) + 1;
        ncols = ncols - map_size(2) + 1;
        
        for irow = 1:nrows
            for jcol = 1:ncols
                
                if caffe == 1
                     patch_data = zeros(map_size(1), map_size(2), N);
                    for i = 1:N
                        patch_data(:,:,i) = img_maps(irow:irow+map_size(1)-1, jcol:jcol+map_size(2)-1, i)';
                    end
                    patch_data = reshape(patch_data, map_size(1)*map_size(2)*N, 1);
                else
                    patch_data = reshape(img_maps(irow:irow+map_size(1)-1, jcol:jcol+map_size(2)-1, :), map_size(1)*map_size(2)*N, 1);
                end
                %patch_data = reshape(img_maps(irow:irow+map_size(1)-1, jcol:jcol+map_size(2)-1, :), map_size(1)*map_size(2)*N, 1);
                
                
                
                % decide the correct patch id
                px = jcol - 1;
                py = irow - 1;
                
                for i = size(offset,1):-1:1
                    ox = offset(i, 1);
                    oy = offset(i, 2);
                    k = offset(i, 3);
                    
                    px = ox + px*k;
                    py = oy + py*k;
                end
                
                px = px + 1;
                py = py + 1;
                
                frags_reunion(:, (py-1)*patch_num(2) + px) = patch_data;
            end
        end
        
    end
end