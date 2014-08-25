

function [tumor_coordinates_in_volume,...
          edema_coordinates_in_volume,...
          healthy_background_coordinates_in_volume,...
          empty_background_coordinates_in_volume,...
          number_tumor_voxels,...
          number_edema_voxels,...
          number_healthy_background_voxels,...
          number_empty_background_voxels ] = get_indices_in_volume_4_labels ( truth, tumor_label, edema_label, healthy_background_label, empty_background_label)

    number_tumor_voxels =              sum(truth(:) == tumor_label);
    number_edema_voxels =              sum(truth(:) == edema_label);
    number_healthy_background_voxels = sum(truth(:) == healthy_background_label);
    number_empty_background_voxels   = sum(truth(:) == empty_background_label);

    tumor_coordinates_in_volume               = zeros(number_tumor_voxels,3);
    edema_coordinates_in_volume               = zeros(number_edema_voxels,3);
    healthy_background_coordinates_in_volume  = zeros(number_healthy_background_voxels,3);
    empty_background_coordinates_in_volume    = zeros(number_empty_background_voxels,3);

    tumor_counter              = 1;
    edema_counter              = 1;
    healthy_background_counter = 1;
    empty_background_counter   = 1;

    for i = 1 : size(truth, 1)
       for j = 1 : size(truth, 2)
            for k = 1 : size(truth, 3)

                if (truth(i,j,k) == tumor_label)

                    tumor_coordinates_in_volume(tumor_counter,1) = i;
                    tumor_coordinates_in_volume(tumor_counter,2) = j;
                    tumor_coordinates_in_volume(tumor_counter,3) = k;

                    tumor_counter = tumor_counter + 1;

                end    

                if (truth(i,j,k) == edema_label)

                    edema_coordinates_in_volume(edema_counter,1) = i;
                    edema_coordinates_in_volume(edema_counter,2) = j;
                    edema_coordinates_in_volume(edema_counter,3) = k;

                    edema_counter = edema_counter + 1;

                end

                if (truth(i,j,k) == healthy_background_label)

                    healthy_background_coordinates_in_volume(healthy_background_counter,1) = i;
                    healthy_background_coordinates_in_volume(healthy_background_counter,2) = j;
                    healthy_background_coordinates_in_volume(healthy_background_counter,3) = k;

                    healthy_background_counter = healthy_background_counter + 1;

                end  
                
                if (truth(i,j,k) == empty_background_label)

                    empty_background_coordinates_in_volume(empty_background_counter,1) = i;
                    empty_background_coordinates_in_volume(empty_background_counter,2) = j;
                    empty_background_coordinates_in_volume(empty_background_counter,3) = k;

                    empty_background_counter = empty_background_counter + 1;

                end  

            end
       end
    end

end
