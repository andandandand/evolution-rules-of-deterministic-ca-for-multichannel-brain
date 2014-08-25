function [ segmentation, strength_map, steps, execution_time ] =...
    ...
    call_cuda_growcut_vezhnevets_multichannel ( channelOne, channelTwo,...
                                                channelThree, channelFour,...
                                                label, strength, max_norm2 )

% get gpu handle and clear its existing memory
g = gpuDevice(1);

reset(g);

g.FreeMemory;

xDim = size (label, 1);
yDim = size (label, 2);
zDim = size (label, 3);

d_channelOne   = gpuArray(channelOne);
d_channelTwo   = gpuArray(channelTwo);
d_channelThree = gpuArray(channelThree);
d_channelFour  = gpuArray(channelFour);


d_strength = gpuArray(single(strength));
d_label    = gpuArray(uint8(label));

d_new_strength = gpuArray(single(strength));
d_new_label    = gpuArray(uint8(label));

%important: convergence-deciding array
d_still_updating = gpuArray(uint8(zeros(xDim, yDim, zDim)));


%create kernels on the gpu and set grid and block size
blockSizeX = 8;
blockSizeY = 8;
blockSizeZ = 8;

gridSizeX = ceil(xDim/blockSizeX);
gridSizeY = ceil(yDim/blockSizeY);
gridSizeZ = ceil(zDim/blockSizeZ);

evol_cells_kernel =...
    parallel.gpu.CUDAKernel('growcut_3D_euclid_4_channels_vezhnevets.ptx',...
                            'growcut_3D_euclid_4_channels_vezhnevets.cu',...
                            '_Z10evol_cellsPKfS0_S0_S0_S0_PKhPfPhS4_iiif');

update_old_step_kernel =...
    parallel.gpu.CUDAKernel('growcut_3D_euclid_4_channels_vezhnevets.ptx',...
                            'growcut_3D_euclid_4_channels_vezhnevets.cu',...
                            '_Z15update_old_stepPKfPfPKhPhiii');



evol_cells_kernel.ThreadBlockSize      = [blockSizeX blockSizeY blockSizeZ];
evol_cells_kernel.GridSize             = [gridSizeX gridSizeY gridSizeZ];

update_old_step_kernel.ThreadBlockSize = [blockSizeX blockSizeY blockSizeZ];
update_old_step_kernel.GridSize        = [gridSizeX gridSizeY gridSizeZ];


tic;

d_still_updating (1,1,1) = 1;

steps=0;

%iterate till full convergence is achieved
while ( nnz(d_still_updating) ~= 0)
    
    [d_new_strength, d_new_label, d_still_updating] =...
        ...
        feval(evol_cells_kernel,...
              ...
              d_channelOne,...
              d_channelTwo,...
              d_channelThree,...
              d_channelFour,...
              d_strength,...
              d_label,...
              d_new_strength,...
              d_new_label,...
              d_still_updating,...
              xDim, yDim, zDim,...
              max_norm2);
    
    
    [d_strength, d_label]=...
        ...
        feval(update_old_step_kernel,...
              d_new_strength,...
              d_strength,...
              d_new_label,...
              d_label,...
              xDim, yDim, zDim);
    
    steps = steps + 1;
    
end
execution_time = toc();

segmentation = gather(d_new_label);
strength_map = gather(d_new_strength);

clear d_new_label d_new_strength d_label d_strength...
      d_channelOne d_channelTwo d_channelThree d_channelFour

end

