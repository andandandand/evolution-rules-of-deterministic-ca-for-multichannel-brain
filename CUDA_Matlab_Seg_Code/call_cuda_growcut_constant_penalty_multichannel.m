function [new_label, new_strength, steps, execution_time] =...
    call_cuda_growcut_constant_penalty_multichannel (   channelOne, channelTwo, channelThree, channelFour,...
                                                        label, strength, max_norm2,...
                                                        ...
                                                        means, sdevs, ndevs,...
                                                        high_penalty, low_penalty,...
                                                        ...
                                                        slice_to_show...
                                                     )

g = gpuDevice(1);

reset(g);

disp ('free memory in gpu ');
g.FreeMemory

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

d_means        = gpuArray(single([0 means'])); %0 added at the start to prevent indexing conflicts cuda <-> matlab
d_sdevs        = gpuArray(single([0 sdevs']));

blockSizeX = 8;
blockSizeY = 8;
blockSizeZ = 8; 

gridSizeX = ceil(xDim/blockSizeX);
gridSizeY = ceil(yDim/blockSizeY);
gridSizeZ = ceil(zDim/blockSizeZ);

evol_cells_kernel      = parallel.gpu.CUDAKernel('growcut_3D_euclid_4_channels_constant_penalty.ptx',...
                                                 'growcut_3D_euclid_4_channels_constant_penalty.cu',...
                                                 '_Z10evol_cellsPKfS0_S0_S0_S0_PKhPfPhS4_S0_S0_fffiiif');

update_old_step_kernel = parallel.gpu.CUDAKernel('growcut_3D_euclid_4_channels_constant_penalty.ptx',...
                                                 'growcut_3D_euclid_4_channels_constant_penalty.cu',...
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
                   high_penalty,...
                   low_penalty,...
                   xDim, yDim, zDim,...
                   max_norm2);
        
         [d_strength, d_label]=...
            ... 
            feval( update_old_step_kernel,...
                   d_new_strength, d_strength,...
                   d_new_label,    d_label,...
                   xDim, yDim, zDim);
         
%          if (update_means)      
%                 % indexing consistency is important here
%                 d_means(2) = mean( d_imdata( (d_new_label == 1) ) ); 
%                 d_sdevs(2) = std ( d_imdata( (d_new_label == 1) ) );
% 
%                 d_means(3) = mean( d_imdata( (d_new_label == 2) ) ); 
%                 d_sdevs(3) = std ( d_imdata( (d_new_label == 2) ) );
% 
%                 d_means(4) = mean( d_imdata( (d_new_label == 3) ) ); 
%                 d_sdevs(4) = std ( d_imdata( (d_new_label == 3) ) );
%          end                
        
         steps = steps + 1;
         
%          if( mod(steps,5) == 0 )
%             figure(steps), imshow ( d_new_label(:, :, slice_to_show), 'DisplayRange', [] );
%          end                 
end
execution_time = toc();  

new_label    = gather (d_new_label);
new_strength = gather (d_new_strength);


end




