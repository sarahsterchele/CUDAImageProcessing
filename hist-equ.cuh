#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H

#define MAX_THREADS 1024

typedef struct
{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    

typedef struct
{
    int w;
    int h;
    unsigned char * img_r;
    unsigned char * img_g;
    unsigned char * img_b;
} PPM_IMG;

typedef struct
{
    int w;
    int h;
    unsigned char * img_y;
    unsigned char * img_u;
    unsigned char * img_v;
} YUV_IMG;

typedef struct
{
    int width;
    int height;
    float * h;
    float * s;
    unsigned char * l;
} HSL_IMG;

__device__ unsigned char GPU_clip_rgb(int x);
__device__ float GPU_Hue_2_RGB( float v1, float v2, float vH );

void GPU_histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void GPU_histogram_equalization(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin);
YUV_IMG GPU_rgb2yuv(PPM_IMG img_in);
YUV_IMG rgb2yuv(PPM_IMG img_in);

PPM_IMG GPU_yuv2rgb(YUV_IMG img_in);
HSL_IMG GPU_rgb2hsl(PPM_IMG img_in);
PPM_IMG GPU_hsl2rgb(HSL_IMG img_in);

/* kernels */

__global__ void GPU_histogram_kernel(int * hist_out, unsigned char * img_in, int nbr_bin, int img_size);
__global__ void GPU_histogram_equalization_kernel(int *cdf_table, unsigned char *img_out, unsigned char * img_in, int img_size, int nbr_bin, int min, int d);
__global__ void GPU_yuv2rgb_kernel(unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *y, unsigned char *u, unsigned char *v, int img_size);
__global__ void GPU_rgb2yuv_kernel(unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *y, unsigned char *u, unsigned char *v, int img_size);
__global__ void GPU_rgb2hsl_kernel(unsigned char *r, unsigned char *g, unsigned char *b, float *h, float *s, unsigned char *l, int img_size);
__global__ void GPU_hsl2rgb_kernel(unsigned char *r, unsigned char *g, unsigned char *b, float *h, float *s, unsigned char *l, int img_size);

#endif