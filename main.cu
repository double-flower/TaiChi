#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include "morphutils.h"
#include "TaiChi.h"

using namespace std;

int M, N;
unsigned long nnz;
int *csrRowIndexHostPtr = 0;
int *csrColIndexHostPtr = 0;
float *csrValHostPtr = 0;
float *xHostPtr = 0;
float *yHostPtr = 0;
char matrixName[1024] = {0};

signed main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("usage: ./exe MatrixName\n");
        return 0;
    }

    char file_name[1024] = {0};
    char temp[1024] = {0};
    strcpy(file_name, argv[1]);
    strcpy(temp, argv[1]);
    char *mtx_pure_name = strrchr(temp, '/');
    int len = strlen(mtx_pure_name);
    mtx_pure_name[len - 4] = '\0';
    strcpy(matrixName, mtx_pure_name+1);

    printf("reading file %s\n", file_name);
    readMtx(file_name, M, N, nnz, csrRowIndexHostPtr, csrColIndexHostPtr, csrValHostPtr);
    printf("M=%d N=%d nnz=%d\n", M, N, nnz);

    auto *neg_format_new = (TaiChi_new *)malloc(sizeof(TaiChi_new));
    memset(neg_format_new, 0, sizeof(TaiChi_new));
    char *matrixPath = file_name;

    // MMSparse partition and format convertion
    partition_shapes(matrixPath, M, N, nnz, csrRowIndexHostPtr, csrColIndexHostPtr, csrValHostPtr, neg_format_new);
       
    if (csrValHostPtr != nullptr)
    {
        free(csrValHostPtr);
        csrValHostPtr = nullptr;
    }
    
    xHostPtr = (float *)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++)
        xHostPtr[i] = 1.0;
    
    yHostPtr = (float *)malloc(sizeof(float) * M);
    memset(yHostPtr, 0.0, sizeof(float) * M);
    
    // correct result
    auto *y_ref = (float *)malloc(sizeof(float) * M);
    for (int i = 0; i < M; i++)
    {
        y_ref[i] = 0.0;
        for (int j = csrRowIndexHostPtr[i]; j < csrRowIndexHostPtr[i + 1]; j++)
        {
            y_ref[i] += xHostPtr[csrColIndexHostPtr[j]];
        }
    }
    
    // spmv calculation
    taichi_SpMV(M, N, neg_format_new, xHostPtr, yHostPtr);
    
    // validate calculated result
    int counter_wrong = 0;
    for (int i = 0; i < M; i++)
    {
        if (abs(yHostPtr[i] - y_ref[i]) > 1e-6)
        {
            if (counter_wrong == 0)
                printf("yHostPtr[%d]=%f, y_ref[%d]=%f\n", i, yHostPtr[i], i, y_ref[i]);
            counter_wrong++;
        }
    }

    printf("Warning: %d are wrong!\n", counter_wrong);

    // free memory
    if (xHostPtr != nullptr)
    {
        free(xHostPtr);
        xHostPtr = nullptr;
    }
    if (yHostPtr != nullptr)
    {
        free(yHostPtr);
        yHostPtr = nullptr;
    }
    if (y_ref != nullptr)
    {
        free(y_ref);
        y_ref = nullptr;
    }

    free_taichi_new(neg_format_new);

    if (csrRowIndexHostPtr != nullptr)
    {
        free(csrRowIndexHostPtr);
        csrRowIndexHostPtr = nullptr;
    }
    if (csrColIndexHostPtr != nullptr)
    {
        free(csrColIndexHostPtr);
        csrColIndexHostPtr = nullptr;
    }

    return 0;
}
