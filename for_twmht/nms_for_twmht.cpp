#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <stdbool.h>
using namespace std;



struct boxes
{
    float x,y,w,h,s;

}typedef box;



float IOUcalc(box b1, box b2)
{
    float ai = (float)(b1.w + 1)*(b1.h + 1);
    float aj = (float)(b2.w + 1)*(b2.h + 1);
//    float ai = (float)(b1.w)*(b1.h);
//    float aj = (float)(b2.w)*(b2.h);
    float x_inter, x2_inter, y_inter, y2_inter;

    x_inter = max(b1.x,b2.x);
    y_inter = max(b1.y,b2.y);

    x2_inter = min((b1.x + b1.w),(b2.x + b2.w));
    y2_inter = min((b1.y + b1.h),(b2.y + b2.h));

//     float w = (float)max((float)0, x2_inter - x_inter);  
//     float h = (float)max((float)0, y2_inter - y_inter); 
    float w = (float)max((float)0, x2_inter - x_inter + 1);  
    float h = (float)max((float)0, y2_inter - y_inter + 1);  

    float inter = ((w*h)/(ai + aj - w*h));
    return inter;
}


void nms_best(box *b, int count, bool *res)
{

    float theta = 0.5;

    for(int i=0; i<count; i++)
    {
        res[i] = true;
    }

    for(int i=0; i<count; i++)
    {

        for(int j=0; j<count; j++)
        {

            if(b[i].s > b[j].s)
            {

                if(IOUcalc(b[i],b[j]) > theta)
                { 
                    res[j] = false; 
                }

            }



        }

    }
    
}


int main()
{
    int count = 75;

    bool res[count];
    box b[count];
    
    std::ifstream in;
    std::string line;

    for(int i = 0; i<count; i++)
    {
        res[i] = false;
    }

    in.open("input_box_dssa56_75.txt"); //x1, y1, w, h
    if (in.is_open()) 
    {
        int i = 0;
        while(getline(in, line))
        {
            istringstream iss(line);
            iss >> b[i].x;
            iss >> b[i].y;
            iss >> b[i].w;
            iss >> b[i].h;
            i+=1;
            if(i==count) break;
        }
    }
    in.close();
    
    in.open("sorted_indices_75.txt");
    if (in.is_open()) 
    {
        int i = 0;
        int cur = -1;
        while(in >> cur)
        {
            b[cur].s = 1 - 0.01*i;
            i+=1;
            if(i==count) break;
        }
    }
    in.close();



    nms_best(b,count,res);

    for(int i = 0; i<count; i++)
    {
        if(res[i])
        {
            printf("%f %f %f %f\n", b[i].x, b[i].y, b[i].w, b[i].h);
        }
    }

    return 0;
}