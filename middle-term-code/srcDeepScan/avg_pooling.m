function z = avg_pooling(x, scale)
    z1 = convn(x, ones(scale) / (scale ^ 2), 'valid'); 
    z = z1(1 : scale : end, 1 : scale : end, :);
end