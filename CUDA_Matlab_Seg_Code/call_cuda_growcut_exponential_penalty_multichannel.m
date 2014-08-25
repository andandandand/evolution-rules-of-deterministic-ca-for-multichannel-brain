function [new_label, new_strength, steps, execution_time] =...
    call_cuda_growcut_exponential_penalty_multichannel (channelOne,...
                                                        channelTwo,...
                                                        channelThree,...
                                                        channelFour,...
                                                        label, strength,...
                                                        max_norm2,...
                                                        means, sdevs,...
                                                        ndevs)

% get gpu handle and reset memory                                                    
g = gpuDevice(1);

reset(g);

g.FreeMemory;

xDim = size(channelOne,1);
yDim = size(channelOne,2);
zDim = size(channelOne,3);

d_still_updating     = gpuArray(uint8(zeros(xDim, yDim, zDim)));

%pass initial data to the gpu 
d_channelOne         = gpuArray(single(channelOne));
d_channelTwo         = gpuArray(single(channelTwo));
d_channelThree       = gpuArray(single(channelThree));
d_channelFour        = gpuArray(single(channelFour));

d_strength     = gpuArray(single(strength));
d_label        = gpuArray(uint8(label));

d_new_strength = gpuArray(single(strength));
d_new_label    = gpuArray(uint8(label));

%0 added at the start to prevent indexing conflicts cuda <-> matlab
d_means        = gpuArray(single([0 means'])); 
d_sdevs        = gpuArray(single([0 sdevs']));

blockSizeX = 8;
blockSizeY = 8;
blockSizeZ = 8; 

gridSizeX = ceil(xDim/blockSizeX);
gridSizeY = ceil(yDim/blockSizeY);
gridSizeZ = ceil(zDim/blockSizeZ);

evol_cells_kernel      = parallel.gpu.CUDAKernel('growcut_3D_euclid_4_channels_exponential_penalty.ptx',...
                                                 'growcut_3D_euclid_4_channels_exponential_penalty.cu',...
                                                 '_Z10evol_cellsPKfS0_S0_S0_S0_PKhPfPhS4_S0_S0_fiiif');

update_old_step_kernel = parallel.gpu.CUDAKernel('growcut_3D_euclid_4_channels_exponential_penalty.ptx',...
                                                 'growcut_3D_euclid_4_channels_exponential_penalty.cu',...
                                                 '_Z15update_old_stepPKfPfPKhPhiii');


evol_cells_kernel.ThreadBlockSize      = [blockSizeX blockSizeY blockSizeZ];
evol_cells_kernel.GridSize             = [gridSizeX gridSizeY gridSizeZ];

update_old_step_kernel.ThreadBlockSize = [blockSizeX blockSizeY blockSizeZ];
update_old_step_kernel.GridSize        = [gridSizeX gridSizeY gridSizeZ];

steps = 0;

%important: iterate till full convergence is achieved
d_still_updating (1,1,1) = 1; 
tic();
while ( nnz (d_still_updating) ~= 0) 
   
         [d_new_strength, d_new_label, d_still_updating] =...
            ...
            feval( evol_cells_kernel,...
                   d_channelOne,...
                   d_channelTwo,...
                   d_channelThree,...
                   d_channelFour,...
                   d_strength,...
                   d_label,...
                   d_new_strength,...
                   d_new_label,...
                   d_still_updating,...
                   d_means, ...
                   d_sdevs,...
                   ndevs,...
                   xDim, yDim, zDim,...
                   max_norm2);
        
         [d_strength, d_label]=...
            ... 
            feval( update_old_step_kernel,...
                   d_new_strength, d_strength,...
                   d_new_label,    d_label,...
                   xDim, yDim, zDim);          
        
         steps = steps + 1;
         
%          if( mod(steps,5) == 0 )
%             figure(steps), imshow ( d_new_label(:, :, slice_to_show), 'DisplayRange', [] );
%          end                 
end

execution_time = toc();  

new_label    = gather (d_new_label);
new_strength = gather (d_new_strength);


end




