%% prep BRATS real data for CIMENICS 2014 tests

clear; close all;

%script to prepare BRATS data for validation 

%generates label space, bounded volume and random seed positions throughout volume 

%% load truth
s = 'browse for truth data'
[truth, info] = ReadData3D;

%% load t1Gad

s = 'browse for t1Gad data'
[t1Gad, info] = ReadData3D;


%% rotate data for proper coronal viewing

truth = imrotate (truth, 90);
t1Gad = imrotate (t1Gad, 90);


%% define values for 4 labels

tumor_label              = 1;
edema_label              = 2;
healthy_background_label = 3;
empty_background_label   = 4;


%%  define brain mask (no empty voxels)
brain_mask = single(t1Gad > 0);

truth = truth + brain_mask;

%figure('name', 'before change slice 55'), imshow(truth(:,:,55),'DisplayRange',[]);


%% change label space in truth 

truth = changem (truth, [tumor_label edema_label healthy_background_label empty_background_label], [3 2 1 0]);

%figure('name', 'after change slice 55'),   imshow(truth(:,:,55), 'DisplayRange',[]);


%% generate bounded truth, obtain lowest and highest indices

[bounded_truth, ... 
 lowest_i,  lowest_j,  lowest_k, ...
 highest_i, highest_j, highest_k] = honest_bounding_box( truth, tumor_label, edema_label );


%% get indices for tumor, edema, background in truth

[tumor_coordinates_in_volume,...
          edema_coordinates_in_volume,...
          healthy_background_coordinates_in_volume,...
          empty_background_coordinates_in_volume,...
          number_tumor_voxels,...
          number_edema_voxels,...
          number_healthy_background_voxels,...
          number_empty_background_voxels ] = get_indices_in_volume_4_labels ( truth, tumor_label, edema_label, healthy_background_label, empty_background_label);


%% generate random list of indices for tumor, edema and background to place seeds on

random_tumor_indices_in_volume              = randperm(number_tumor_voxels);
random_edema_indices_in_volume              = randperm(number_edema_voxels);
random_healthy_background_indices_in_volume = randperm(number_healthy_background_voxels);
random_empty_background_indices_in_volume   = randperm(number_empty_background_voxels);


%% get slice (index) with most tumor cells in the volume 
top_tumor_cells_in_slice  = 1;
tumor_slice               = 1;

for i = 1 : size(truth,3)

    logical_tumor_slice  = (truth(:,:,i)== tumor_label);
    tumor_cells_in_slice = sum(logical_tumor_slice(:));
    if (tumor_cells_in_slice > top_tumor_cells_in_slice)
        top_tumor_cells_in_slice = tumor_cells_in_slice;
        tumor_slice = i;
    end
end
figure('name', 'tumor slice - t1Gad'), imshow(t1Gad(:,:,tumor_slice), 'DisplayRange',[]);
figure('name', 'tumor slice - truth'), imshow(truth(:,:,tumor_slice), 'DisplayRange',[]);

%% clear temporal vars of script before saving curated data 

clearvars top_tumor_cells logical_tumor_slice tumor_cells_in_slice s i;