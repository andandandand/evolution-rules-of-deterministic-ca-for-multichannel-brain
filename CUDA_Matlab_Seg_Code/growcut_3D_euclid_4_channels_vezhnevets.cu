// compile with: nvcc growcut_3D_euclid_4_channels_vezhnevets.cu -arch sm_20 -ptx 
// info on thread indices: http://stackoverflow.com/questions/7318002/3d-image-indices

__global__ void update_old_step ( const float          *d_new_strength,
                                  float				       *d_strength,
                                  const unsigned char     *d_new_label,							
                                  unsigned char				  *d_label,

                                  const int                      xDim, 
                                  const int                      yDim,
                                  const int                      zDim)

{	
	  //important: x = col, y = row
	  //xDim = col_dim, yDim = row_dim
	  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	  unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

	  if ( (x >= xDim) || (y >= yDim) || (z >= zDim) ) return;
    
	  unsigned int offset = x + (xDim * y) + (xDim *yDim * z);

	  //if the thread id expressed in offset is out of bounds, abort thread execution
	   int nnodes = xDim * yDim * zDim;
	   
	   if (offset >= nnodes) return; 

       //update old step
	   d_strength[offset] = d_new_strength[offset];	
	   d_label[offset] = d_new_label[offset];

}


__global__ void evol_cells(  const float  			  		   *d_channelOne,
							 const float  			  		   *d_channelTwo,
							 const float                     *d_channelThree,
							 const float                      *d_channelFour,
                             
                             const float 					     *d_strength,
						     const unsigned char 			        *d_label,
                             	   
                             	   float 				     *d_new_strength,
							 	   unsigned char 		        *d_new_label,
							       unsigned char 	       *d_still_updating,
							
                             const int xDim, 
							 const int yDim, 
							 const int zDim,
							 const float max_I)
{	  
  
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

  if ( (x >= xDim) || (y >= yDim) || (z >= zDim) )
  {return;}
    
  unsigned int offset = x + (xDim * y) + (xDim * yDim * z);

   //3D Von Neumann neighborhood
   unsigned int left   = (x - 1) + (xDim * y) + (xDim * yDim * z);
   unsigned int right  = (x + 1) + (xDim * y) + (xDim * yDim * z);
   unsigned int top    = x + (xDim * (y - 1)) + (xDim * yDim * z);
   unsigned int bottom = x + (xDim * (y + 1)) + (xDim * yDim * z);
   unsigned int front  = x + (xDim * y) + (xDim * yDim * (z - 1));
   unsigned int back   = x + (xDim * y) + (xDim * yDim * (z + 1));


   int neigh          = -1;
   
   float gfunc        = -1.0f;
   float eval_product = -1.0f;
   
   //will change to 1 if the cell is overpowered by an attacker
   d_still_updating[offset] = 0;
   
   float two = 2.0;
   //check left neighbor
   if (x != 0){
	   
	   neigh = left;
      
	   /*begin eval region*/
	   gfunc = 1.0f - ( sqrt ( 

	   					pow( (d_channelOne[offset] - d_channelOne[neigh]), two ) 
	   											   +
	   	                pow( (d_channelTwo[offset] - d_channelTwo[neigh]), two )
	   	                						   +
	   	                pow( (d_channelThree[offset] - d_channelThree[neigh]), two )
	   	                						   +
	   	                pow( (d_channelFour[offset] - d_channelFour[neigh]), two )
	   					
	   					)

	               / max_I);

	   eval_product = gfunc * d_strength[neigh];

	   //if the current cell is overpowered 
	   if( ( eval_product > d_strength[offset])){
	   
		   d_new_strength[offset] = eval_product;
		   d_new_label[offset] = d_label[neigh];
	       
	       d_still_updating[offset] = 1;
       
	   }
	   /*end eval region*/
   }

   //check right neighbor
   if (x != (xDim - 1)){
		
		neigh = right;
		
	   /*begin eval region*/
	   gfunc = 1.0f - ( sqrt ( 

	   					pow( (d_channelOne[offset] - d_channelOne[neigh]), two ) 
	   											   +
	   	                pow( (d_channelTwo[offset] - d_channelTwo[neigh]), two )
	   	                						   +
	   	                pow( (d_channelThree[offset] - d_channelThree[neigh]), two )
	   	                						   +
	   	                pow( (d_channelFour[offset] - d_channelFour[neigh]), two )
	   					
	   					)

	               / max_I);

	   eval_product = gfunc * d_strength[neigh];

	   //if the current cell is overpowered 
	   if( ( eval_product > d_strength[offset])){
	   
		   d_new_strength[offset] = eval_product;
		   d_new_label[offset] = d_label[neigh];
	       
	       d_still_updating[offset] = 1;
       
	   }
	   /*end eval region*/
   }

   //check top neighbor
   if(y != 0){
	   
	   neigh = top;

	   /*begin eval region*/
	   gfunc = 1.0f - ( sqrt ( 

	   					pow( (d_channelOne[offset] - d_channelOne[neigh]), two ) 
	   											   +
	   	                pow( (d_channelTwo[offset] - d_channelTwo[neigh]), two )
	   	                						   +
	   	                pow( (d_channelThree[offset] - d_channelThree[neigh]), two )
	   	                						   +
	   	                pow( (d_channelFour[offset] - d_channelFour[neigh]), two )
	   					
	   					)

	               / max_I);

	   eval_product = gfunc * d_strength[neigh];

	   //if the current cell is overpowered 
	   if( ( eval_product > d_strength[offset])){
	   
		   d_new_strength[offset] = eval_product;
		   d_new_label[offset] = d_label[neigh];
	       
	       d_still_updating[offset] = 1;
       
	   }
	   /*end eval region*/
		
   }

   //check bottom neighbor
   if (y != (yDim - 1)){
		
		neigh = bottom;
		
	   /*begin eval region*/
	   gfunc = 1.0f - ( sqrt ( 

	   					pow( (d_channelOne[offset] - d_channelOne[neigh]), two ) 
	   											   +
	   	                pow( (d_channelTwo[offset] - d_channelTwo[neigh]), two )
	   	                						   +
	   	                pow( (d_channelThree[offset] - d_channelThree[neigh]), two )
	   	                						   +
	   	                pow( (d_channelFour[offset] - d_channelFour[neigh]), two )
	   					
	   					)

	               / max_I);

	   eval_product = gfunc * d_strength[neigh];

	   //if the current cell is overpowered 
	   if( ( eval_product > d_strength[offset])){
	   
		   d_new_strength[offset] = eval_product;
		   d_new_label[offset] = d_label[neigh];
	       
	       d_still_updating[offset] = 1;
       
	   }
	   /*end eval region*/
   }

   //check front neighbor
   if(z != 0){
		
		neigh = front;

	   /*begin eval region*/
	   gfunc = 1.0f - ( sqrt ( 

	   					pow( (d_channelOne[offset] - d_channelOne[neigh]), two ) 
	   											   +
	   	                pow( (d_channelTwo[offset] - d_channelTwo[neigh]), two )
	   	                						   +
	   	                pow( (d_channelThree[offset] - d_channelThree[neigh]), two )
	   	                						   +
	   	                pow( (d_channelFour[offset] - d_channelFour[neigh]), two )
	   					
	   					)

	               / max_I);

	   eval_product = gfunc * d_strength[neigh];

	   //if the current cell is overpowered 
	   if( ( eval_product > d_strength[offset])){
	   
		   d_new_strength[offset] = eval_product;
		   d_new_label[offset] = d_label[neigh];
	       
	       d_still_updating[offset] = 1;
       
	   }
	   /*end eval region*/
		
   }

   //check back neighbor
   if(z != (zDim - 1)){
		
		neigh=back;

	   /*begin eval region*/
	   gfunc = 1.0f - ( sqrt ( 

	   					pow( (d_channelOne[offset] - d_channelOne[neigh]), two ) 
	   											   +
	   	                pow( (d_channelTwo[offset] - d_channelTwo[neigh]), two )
	   	                						   +
	   	                pow( (d_channelThree[offset] - d_channelThree[neigh]), two )
	   	                						   +
	   	                pow( (d_channelFour[offset] - d_channelFour[neigh]), two )
	   					
	   					)

	               / max_I);

	   eval_product = gfunc * d_strength[neigh];

	   //if the current cell is overpowered 
	   if( ( eval_product > d_strength[offset])){
	   
		   d_new_strength[offset] = eval_product;
		   d_new_label[offset] = d_label[neigh];
	       
	       d_still_updating[offset] = 1;
       
	   }
	   /*end eval region*/
		
	}
}


	
