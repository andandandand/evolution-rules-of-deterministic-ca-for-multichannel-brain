%% prep BRATS real data for CIMENICS 2014 tests

clear; close all;

%script to prepare BRATS data for validation 

%generates label space, bounded volume and random seed positions throughout volume 

numberOfFiles = 10;
for fileIndex = 1 : numberOfFiles

%% load truth
% s = 'browse for truth data'
% [truth, info] = ReadData3D;

%complete truth file path and name

if (fileIndex < 10)
    truth_file_path =...
        ['C:\Users\antonio\Dropbox\Tesis de Antonio Rueda\data sets (Brats y Tumorsim)\Brats 2012 - Real Brains + Truth\Sim_Brains\Brats sim truth\SimBRATS_HG000'...
        num2str(fileIndex) '_complete_truth.mha'];
else
    truth_file_path =...
        ['C:\Users\antonio\Dropbox\Tesis de Antonio Rueda\data sets (Brats y Tumorsim)\Brats 2012 - Real Brains + Truth\Sim_Brains\Brats sim truth\SimBRATS_HG00'...
        num2str(fileIndex) '_complete_truth.mha'];
end
truth_info      = mha_read_header(truth_file_path);
truth           = mha_read_volume(truth_info);

clear truth_file_path

%% load t1Gad
if (fileIndex < 10)
t1Gad_path = ['C:\Users\antonio\Dropbox\Tesis de Antonio Rueda\data sets (Brats y Tumorsim)\Brats 2012 - Real Brains + Truth\Sim_Brains\Brats sim brains\SimBRATS_HG000'...
               num2str(fileIndex) '\SimBRATS_HG000' num2str(fileIndex) '_T1C.mha'];
else
t1Gad_path = ['C:\Users\antonio\Dropbox\Tesis de Antonio Rueda\data sets (Brats y Tumorsim)\Brats 2012 - Real Brains + Truth\Sim_Brains\Brats sim brains\SimBRATS_HG00'...
               num2str(fileIndex) '\SimBRATS_HG00' num2str(fileIndex) '_T1C.mha'];    
end
          
t1Gad_info = mha_read_header(t1Gad_path);
t1Gad      = mha_read_volume(t1Gad_info);

clear t1Gad_path

%% rotate data for proper coronal viewing

truth = imrotate (truth, 90);
t1Gad = imrotate (t1Gad, 90);


%% define values for 4 labels

tumor_label              = 1;
edema_label              = 2;
healthy_background_label = 3;
empty_background_label   = 4;



%% change label space in truth 
% original: tumor = 5, edema = 4, empty background = 0 
truth = changem (truth, [tumor_label edema_label healthy_background_label healthy_background_label healthy_background_label healthy_background_label empty_background_label],...
                        [5           4           1                        2                        3                        6                        0                     ]);

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

%% concatenate path and fileName

save_path = 'C:\Users\antonio\Dropbox\Tesis de Antonio Rueda\CIMENICS\Test Datasets';

if fileIndex < 10
    fileName = ['\SimBRATS_HG000' num2str(fileIndex) '_T1Gad_4_labels.mat'];
else
    fileName = ['\SimBRATS_HG00'  num2str(fileIndex) '_T1Gad_4_labels.mat'];
end

path_and_fileName = [save_path fileName];
save numberOfFiles fileIndex
clear numberOfFiles fileIndex
save(path_and_fileName);
load numberOfFiles fileIndex
end

