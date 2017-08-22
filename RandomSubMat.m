function [ SubMat, SubMat_Idx, A_red, A_red_Idx ] = RandomSubMat( A, Size, Idx )
%Creates a random matrix from 'Size' randomly selected columns of 'A'.
%Outputs the random matrix, ordered column indices, and the remaining
%columns of A as a separate matrix.

New = [];
New_Idx = [];
for i=1:Size,
    n = size(A,2);
    col = randi(n);
    New_Idx = [New_Idx Idx(col)];
    New = [New A(:,col)];
    A(:,col) = [];
    Idx(col) = [];
end
SubMat = New;
SubMat_Idx = New_Idx;
A_red = A;
A_red_Idx = Idx;
end

