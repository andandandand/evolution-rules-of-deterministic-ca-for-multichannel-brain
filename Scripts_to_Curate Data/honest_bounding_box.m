function [bounded_volume, ... 
          lowest_i,  lowest_j, lowest_k, ...
          highest_i, highest_j, highest_k] = honest_bounding_box( volume, tumor_label, edema_label )

      
        lowest_i = 999999;
        lowest_j = 999999;
        lowest_k = 999999;

        highest_i = -999999;
        highest_j = -999999;
        highest_k = -999999;


        xDim = size(volume,1);
        yDim = size(volume,2);
        zDim = size(volume,3);

for i=1 : xDim
   for j=1 : yDim
       for k=1 : zDim
      
           if ( (volume(i,j,k) == tumor_label) || (volume(i,j,k) == edema_label))
           
               if (i < lowest_i)
                   lowest_i = i;
               end
           
               if (j < lowest_j)
                   lowest_j = j;
               end
           
               if (k < lowest_k)
                   lowest_k = k;
               end
               
               if (i > highest_i)
                   highest_i = i;
               end
           
               if (j > highest_j)
                   highest_j = j;
               end
               
               if (k > highest_k)
                   highest_k = k;
               end
               
           end
           
       end
   end
end

bounded_volume = volume(lowest_i:highest_i, lowest_j:highest_j, lowest_k:highest_k);


end

