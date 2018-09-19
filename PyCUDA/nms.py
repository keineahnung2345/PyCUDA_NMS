import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
import pycuda.driver as drv
import string
import tensorflow as tf
import time
import math

template = """
#define THETA $THETA
#include <stdio.h>

__device__
float IOUcalc(float* b1, float* b2)
{
    //y1, x1, h, w
    float ai = (float)b1[3]*b1[2];
    float aj = (float)b2[3]*b2[2];
    float x_inter, x2_inter, y_inter, y2_inter;

    x_inter = max(b1[1],b2[1]);
    y_inter = max(b1[0],b2[0]);

    x2_inter = min((b1[1] + b1[3]),(b2[1] + b2[3]));
    y2_inter = min((b1[0] + b1[2]),(b2[0] + b2[2]));
    
    float w = (float)max((float)0, x2_inter - x_inter);  
    float h = (float)max((float)0, y2_inter - y_inter);  

    float inter = ((w*h)/(ai + aj - w*h));
    return inter;
}

__global__
void NMS_GPU(float *d_b, bool *d_res)
{
    int abs_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int abs_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if(d_b[abs_x*5+4] < d_b[abs_y*5+4])
        {
            float* b1 = &d_b[abs_y*5];
            float* b2 = &d_b[abs_x*5];
            
            float iou = IOUcalc(b1,b2);
            
            if(iou>THETA)
            {
                d_res[abs_x] = false; 
            }
        }
}
"""

def to_yxhw(boxes):
    # box format: [y1,x1,y2,x2] to [y1,x1,h,w]
    boxes[:,2:] = boxes[:,2:]-boxes[:,0:2]
    return boxes

def cuda_nms(modules, boxes, scores, yxhw=False):
    if not yxhw:
        boxes = to_yxhw(boxes)
    
    #Prepare data for nms on GPU
    #After this,
    # boxes becomes: [y1,x1,y2,x2,score]
    # results becomes: [True,...]
    count = boxes.shape[0]
    boxes = np.hstack((boxes, np.expand_dims(scores,axis=1)))
    results = np.array([True]*count, dtype=np.bool)
    
    # Perform nms on GPU
    count = boxes.shape[0]
    NMS_GPU = modules.get_function("NMS_GPU")
    #use drv.InOut instead of drv.Out so the value of results can be passed in
    
    #Setting1:works only when count<=1024
    #grid_size, block_size = (1,count,1), (count,1,1)
    
    #Setting2:works when count>1024
    #grid_size, block_size = (count,count,1), (1,1,1)
    
    #Setting3:works when count>1024, faster then Setting2
    block_len = 32
    grid_len = math.ceil(count/block_len)
    grid_size, block_size = (grid_len,grid_len,1), (block_len,block_len,1)
    
    NMS_GPU(drv.In(boxes), drv.InOut(results),
            grid=grid_size, block=block_size)
    return list(np.where(results)[0])
    
if __name__ == "__main__":
    THETA=0.7
    
    #y1, x1, y2, x2
    boxes = np.array([[261, 265, 443, 401],
           [263, 268, 448, 402],
           [268, 267, 452, 400],
           [267, 267, 446, 402],
           [264, 264, 452, 400],
           [264, 267, 442, 401],
           [268, 270, 460, 404],
           [269, 257, 439, 421],
           [268, 268, 443, 399]], dtype=np.float32)
    
    scores = np.array([0.9994855 , 0.999233  , 0.99853265, 0.99948454, 0.99948066,
           0.9998466 , 0.99871445, 0.99285066, 0.95467633], dtype=np.float32)
    
    template = string.Template(template)
    template = template.substitute(THETA=THETA)
    modules = SourceModule(template) 
    # python function will change array's value, so use .copy()
    cuda_start = time.time()
    cuda_results = cuda_nms(modules, boxes.copy(), scores.copy())
    cuda_end = time.time()
    print("CUDA results:", cuda_results)
    print("CUDA version takes {} seconds".format(cuda_end-cuda_start))
    
    tf_results = tf.image.non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size=boxes.shape[0],
        iou_threshold=THETA,
        score_threshold=float('-inf'),
        name=None
    )
    with tf.Session() as sess:
        tf_start = time.time()
        tf_results = sess.run(tf_results)
        tf_end = time.time()
    print("TF results:", tf_results)
    print("TF version takes {} seconds".format(tf_end-tf_start))
    print("CUDA version is {} times faster than TF version!".format((tf_end-tf_start)/(cuda_end-cuda_start)))