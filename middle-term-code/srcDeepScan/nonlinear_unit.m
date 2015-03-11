function X = nonlinear_unit(P, type)
    switch type
        case 1 % sigm
            X = sigm(P);
        case 2 % tanh
            X = tanh_opt(P);
        case 3 % rectifier
            X = rectifier(P);
        case 4 % softmax
            X = softmax(P);
        case 0 % linear
            X = P;
    end
end