function [tp, tn, fp, fn] = validate_n_classes_3D( seg, truth, nclasses )
%function to validate segmentations of n classes
%seg and truth must share the same label space for this to work 

tp = zeros(nclasses,1);
tn = zeros(nclasses,1);
fp = zeros(nclasses,1);
fn = zeros(nclasses,1);

xDim = size(seg,1);
yDim = size(seg,2);
zDim = size(seg,3);

for j = 1 : yDim 
    for i = 1 : xDim
        for k = 1 : zDim            
            for c = 1 : nclasses
       
            %true positive
            if (seg(i,j,k) == c) && (truth(i,j,k) == c)
                
                tp(c) = tp(c) + 1;
            
            end
            
            %true negative
            if (seg(i,j,k) ~= c) && (truth(i,j,k) ~= c)
                
                tn(c) = tn(c) + 1;
            
            end
            
            %false positive
            if (seg(i,j,k) == c) && (truth(i,j,k) ~= c)
                
                fp(c) = fp(c) + 1;
            
            end
            
            %false negative
            if (seg(i,j,k) ~= c) && (truth(i,j,k) == c)
                
                fn(c) = fn(c) + 1;
            
            end
            
            end      
        end
    end
end

end

