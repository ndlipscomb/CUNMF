function [ Wout, Hout, Vperm, Rel_Err, Idx_out] = CUNMF( V, k, varargin )
% Produces a solution to the NMF problem V = WH by means of the
% continuously updating nonnegative matrix factorisation approach
% proposed by Lipscomb, Chang, Chen, & Wang. Rearrangement of the columns 
% of V is considered acceptable.
% This algorithm incorporates the 'nnmf' function from MATLAB. Please refer
% to the documentation here: https://www.mathworks.com/help/stats/nnmf.html
%INPUTS--------------------------------------------------------------------
% V :               The matrix we seek to perform continuously updating 
%                   nonnegative matrix factorisation on. We refer to 'm' as 
%                   the number of rows of 'V' and n as the number of 
%                   columns of 'V'. Column permutations of 'V' will occur
%                   as part of the CUNMF algorithm.
% k :               The rank of the factorisation.
% The following name-value pairs can be used as additional arguments. See
% below.
% ALGORITHM :       Specify whether to use the alternating least squares
%                   'als' algorithm or the multiplicative update 'mult'
%                   algorithm. Default is 'mult'.
% INIT_SIZE :       The size of the initial submatrix of V we perform NMF
%                   and then add to via the continuous update rule. Must be
%                   a positive integer smaller than 'n'. Default is
%                   approximately 10 percent of the columns.
% UPDATE_SIZE :     The size of each additional batch of columns added to
%                   the initial submatrix and 'H' as part of the
%                   continuously updating rule. Must be a positive integer
%                   smaller than 'n' minus INIT_SIZE. Default is 1.
% INIT_ITER :       The maximum number of NMF iterations used for the
%                   factorisation of the initial submatrix of V. 
%                   Default = 10 per column.
% UPDATE_ITER :     The maximum number of NMF iterations used for the
%                   factorisation for each update after the initialisation. 
%                   Default = 10 per column.
% RES_TOL :         NMF termination tolerance on change in size of the 
%                   residual. Default is 0.0001.
% REL_TOL :         NMF termination tolerance on relative change in the 
%                   elements of W and H. Default is 0.0001.
% NUM_TESTS :       The number of random test initialisations that are
%                   factorised in order to produce a good starting point
%                   for the overall problem. Must be a positive integer.
%                   Default is 10.
% W_INIT :          Allows the specification of the factor "W" for the 
%                   initial submatrix's factorisation. Must be 'm' by 'k'.
%                   Default is randomly generated.
% H_INIT :          Allows the specification of the factor "H" for the
%                   initial submatrix's factorisation. Must be 'k' by
%                   INIT_SIZE. Default is randomly generated.
% SEED :            Set the seed for random number generation. Must be a
%                   nonnegative integer argument. Default is randomly
%                   generated.


%OUTPUTS-------------------------------------------------------------------
% Wout :            The factor 'W' (basis matrix) resulting from CUNMF.
% Hout :            The factor 'H' (weight matrix) resulting from CUNMF.
% Vperm :           The column permutated matrix V reconstructed from the
%                   continuous updating approach.
% Rel_Err :         The relative error expressed as 
%                   |Vperm-Wout*Hout|/|Vperm| with respect to the Frobenius
%                   norm resulting from the CUNMF.
% Idx_out :     The ordered list of indices from the original matrix,
%                   V, corresponding to the column permutated matrix Vperm.

%Set up constants and counters.
    [m,n] = size(V);
    V_red = V;
    Idx_orig = linspace(1,n,n);
    Idx_new = [];
    
% Default configuration (not including matrix initialisations).
    par.algorithm = 'mult';
    par.batch_size_I = round(n/10,0)+1;
    par.batch_size_C = 1;
    par.init_iter = 10*par.batch_size_I;
    par.update_iter = 10*par.batch_size_C;
    par.res_TOL = 0.0001;
    par.rel_TOL = 0.0001;
    par.num_tests = 10;
    W = rand(m,k);
    H = rand(k,par.batch_size_I);
    par.seed = randi(2^32-1);
    rng(par.seed); %Set randomly generated seed.
    
% Read optional parameters
    if (rem(length(varargin),2)==1)
        error('Optional parameters must be stated pairwise');
    else
        for i=1:2:(length(varargin)-1)
            switch upper(varargin{i})
                case 'ALGORITHM',           par.algorithm = varargin{i+1};
                case 'INIT_SIZE',           par.batch_size_I = varargin{i+1}; H = rand(k,par.batch_size_I); par.init_iter = 10*par.batch_size_I;
                case 'UPDATE_SIZE',         par.batch_size_C = varargin{i+1}; par.update_iter = 10*par.batch_size_C;
                case 'INIT_ITER',           par.init_iter = varargin{i+1};
                case 'UPDATE_ITER',         par.update_iter = varargin{i+1};
                case 'RES_TOL',             par.res_TOL = varargin{i+1};
                case 'REL_TOL',             par.rel_TOL = varargin{i+1};
                case 'NUM_TESTS',           par.num_tests = varargin{i+1};
                case 'W_INIT',              W = varargin{i+1};
                case 'H_INIT',              H = varargin{i+1};
                case 'SEED',                par.seed = varargin{i+1};
                otherwise
                    error(['Unrecognised argument: ',varargin{i}]);
            end
        end
    end
%Set statset list for initialisation.
opt = statset('MaxIter',par.init_iter,'TolFun',par.res_TOL,'TolX',...
    par.rel_TOL);
%Set the seed.
rng(par.seed);
    
%Create 'num_tests' initialisations. Test which factorisation produces
%the least relative error, then use that as your starting point for the
%continuous updating.
   Err_test = 10^9;
   for i=1:par.num_tests,
       [SubMat, SubMat_Idx, V_red, V_red_Idx] = RandomSubMat(V,...
           par.batch_size_I, Idx_orig);
       [W_ini,H_ini,Err] = nnmf(SubMat,k,'w0',W,'h0',H,'algorithm',...
           par.algorithm,'options',opt);
       if Err <= Err_test,
           W = W_ini; H = H_ini;
           GrowMat = SubMat; V_rem = V_red;
           Idx_red = V_red_Idx; Idx_new = SubMat_Idx;
           Err_test = Err;
       end
   end
%We now have the initialisation matrix 'GrowMat', the reduced matrix
%'V_rem' containing the remaining columns of 'V', and a starting NMF
%with 'W' and 'H'.

%Set statset list for update iterations.
opt = statset('MaxIter',par.update_iter,'TolFun',par.res_TOL,'TolX',...
    par.rel_TOL);

%Begin the continuously updating aspect of the algorithm. We will add
%new randomly selected columns to 'GrowMat' and add their projections to
%'H'.
   while size(V_rem,2) >= par.batch_size_C,
       [New_Batch, SubMat_Idx, V_rem, V_rem_Idx] = RandomSubMat(V_rem,...
           par.batch_size_C, Idx_red);
       Idx_new = [Idx_new SubMat_Idx]; Idx_red = V_rem_Idx;
       GrowMat = [GrowMat New_Batch];
       for i=1:size(New_Batch,2),
           for j=1:k,
               Batch_Proj(j,i) = (transpose(New_Batch(:,i))*W(:,j))/...
                   (transpose(W(:,j))*W(:,j));
           end
       end
       H = [H Batch_Proj];
       [W,H,Err] = nnmf(GrowMat,k,'w0',W,'h0',H,'algorithm',...
           par.algorithm,'options',opt);
   end
%For any remaining columns less than the batch size, perform one last
%projection and NMF iteration.
   if size(V_rem) > 0,
       [Final_Batch, SubMat_Idx, V_rem, V_rem_Idx] = RandomSubMat(V_rem,...
           size(V_rem,2), Idx_red);
       GrowMat = [GrowMat Final_Batch];
       Idx_new = [Idx_new SubMat_Idx];
       for i=1:size(Final_Batch,2),
           for j=1:k,
               Rem_Proj(j,i) = (transpose(Final_Batch(:,i))*W(:,j))/...
                   (transpose(W(:,j))*W(:,j));
           end
       end
       H = [H Rem_Proj];
       [W,H,Err] = nnmf(GrowMat,k,'w0',W,'h0',H,'algorithm',...
           par.algorithm,'options',opt);
   end

%Output the final results.
   Wout = W;
   Hout = H; 
   Vperm = GrowMat;
   Rel_Err = norm(Vperm-Wout*Hout,'fro')/norm(Vperm,'fro');
   Idx_out = Idx_new;
end

