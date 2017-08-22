function [ Wout, Hout, iter, RelErr, Converge ] = BasicNMF( V, W, H, TOL, max_iter )
%Basic NMF algorithm based on Lee & Seung 2001 paper.

[m,n] = size(V); k = size(W,2);

count = 0;
epsi = 10^(-9);
Converge = false;

while count < max_iter,
    H = H.*(transpose(W)*V)./(transpose(W)*W*H+epsi);
    W = W.*(V*transpose(H))./(W*H*transpose(H)+epsi);
    count = count+1;
    RelErr = norm(V-W*H,'fro')/norm(V,'fro');
    if RelErr <= TOL,
        Converge = true;
        break
    end     
end
iter = count;
Wout = W; Hout = H;

end

