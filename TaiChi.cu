#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include <stdlib.h> 
#include <iostream>
#include <fstream>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <cfloat>
#include <errno.h>
#include "cusparse.h"
#include "mmio.h"
#include <dirent.h>
#include <sys/time.h>
#include "utils.h"
#include "TaiChi.h"
#include <cusp/system/cuda/arch.h>
#include "cuda_profiler_api.h"
#include "morphutils.h"

using namespace std;

extern char matrixName[];

#define RECORD_NNZ_DENSITY 0
#define TIMING 1

#define FUNC1 \
    sum += (sign + sign - 1) * x[colIdx];

#define FUNC2 \
    if (sign == 1)              \
        sum += x[colIdx];       \
    else                        \
        sum -= x[colIdx];

#ifdef RECORD_NNZ_DENSITY
    float nnz_density = 0.0;
#endif

inline void checkcuda(cudaError_t result)
{
	if (result != cudaSuccess) {
		printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		printf("hello");
	}
}

inline void checkcusparse(cusparseStatus_t result)
{
	if(result != CUSPARSE_STATUS_SUCCESS){
		printf("CUSPARSE Error, error_code =  %d\n", result);
	}
}

double average(int n, double *data)
{
    double ave = 0.0;
    for(int i = 0; i < n; i++)
        ave += data[i];
    
    return ave / n;
}

double variance(int n, double ave, double *data)
{
    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double temp = data[i] - ave;
        var += (temp * temp);
    }
    
    return var / n;
}

void free_taichi_new(TaiChi_new *taichi)
{
    if (taichi->xStart != NULL) {
        free(taichi->xStart);
        taichi->xStart = NULL;
    }
    if (taichi->yStart != NULL) {
        free(taichi->yStart);
        taichi->yStart = NULL;
    }
    if (taichi->xStop != NULL) {
        free(taichi->xStop);
        taichi->xStop = NULL;
    }

    if (taichi->nDiasPtr != NULL) {
        free(taichi->nDiasPtr);
        taichi->nDiasPtr = NULL;
    }
    if (taichi->neg_offsets != NULL) {
        free(taichi->neg_offsets);
        taichi->neg_offsets = NULL;
    }
    if (taichi->dense_nnz > 0)  {
        if (taichi->csrRow != NULL) {
            free(taichi->csrRow);
            taichi->csrRow = NULL;
        }
        if (taichi->csrCol != NULL) {
            free(taichi->csrCol);
            taichi->csrCol = NULL;
        }
    }
    else  {
        taichi->csrRow = NULL;
        taichi->csrCol = NULL;
    }
    if (taichi != NULL)  {
        free(taichi);
        taichi = NULL;
    }
}

int readMtx(char *filename, int &m, int &n, unsigned long &nnzA, 
            int *&csrRowPtrA, int *&csrColIdxA, float *&csrValA)
{
	int ret_code = 0;
	MM_typecode matcode;

	FILE *f = NULL;
	unsigned long nnzA_mtx_report = 0;
	int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;
	// load matrix
	if ((f = fopen(filename, "r")) == NULL)
		return -1;

	if (mm_read_banner(f, &matcode) != 0) {
		printf("Could not process Matrix Market banner.\n");
		return -2;
	}

	if (mm_is_complex(matcode)) {
		printf("Sorry, data type 'COMPLEX' is not supported. \n");
		return -3;
	}

	if (mm_is_pattern(matcode)) {
		isPattern = 1; 
        // printf("type = Pattern.\n");
	}

	if (mm_is_real(matcode)) {
		isReal = 1; 
        // printf("type = real.\n");
	}

	if (mm_is_integer(matcode)) {
		isInteger = 1; 
        // printf("type = integer.\n");
	}

	ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
	if (ret_code != 0)
		return -4;

	if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode)) {
		isSymmetric = 1;
		// printf("symmetric = true.\n");
	}
	else {
		// printf("symmetric = false.\n");
	}

	int *csrRowPtrA_counter = (int *)malloc((m + 1) * sizeof(int));
	memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));

	int *csrRowIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
	memset(csrRowIdxA_tmp, 0, nnzA_mtx_report * sizeof(int));
	int *csrColIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
	memset(csrColIdxA_tmp, 0, nnzA_mtx_report * sizeof(int));
	float *csrValA_tmp = (float *)malloc(nnzA_mtx_report * sizeof(float));
	memset(csrValA_tmp, 0.0, nnzA_mtx_report * sizeof(float));

	for (unsigned long i = 0; i < nnzA_mtx_report; i++)
	{
		int idxi = 0, idxj = 0;
		double fval = 0.0;
		int ival = 0;

		if (isReal)
			fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
		else if (isInteger) {
			fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
			fval = ival;
		}
		else if (isPattern) {
			fscanf(f, "%d %d\n", &idxi, &idxj);
			fval = 1.0;
		}

		// adjust from 1-based to 0-based
		idxi--;
		idxj--;

		csrRowPtrA_counter[idxi]++;
		csrRowIdxA_tmp[i] = idxi;
		csrColIdxA_tmp[i] = idxj;
		csrValA_tmp[i] = fval;
	}

	if (f != stdin)
		fclose(f);	

	if (isSymmetric) {
		for (unsigned long i = 0; i < nnzA_mtx_report; i++) 
        {
			if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
				csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
		}
	}

	// exclusive scan for csrRowPtrA_counter
	int old_val = 0, new_val = 0;

	old_val = csrRowPtrA_counter[0];
	csrRowPtrA_counter[0] = 0;
	for (unsigned long i = 1; i <= m; i++)
	{
		new_val = csrRowPtrA_counter[i];
		csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i - 1];
		old_val = new_val;
	}

	nnzA = csrRowPtrA_counter[m];
	csrRowPtrA = (int *)malloc((m + 1) * sizeof(int));
	memcpy(csrRowPtrA, csrRowPtrA_counter, (m + 1) * sizeof(int));
	memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));

	csrColIdxA = (int *)malloc(nnzA * sizeof(int));
	memset(csrColIdxA, 0, nnzA * sizeof(int));
	csrValA = (float *)malloc(nnzA * sizeof(float));
	memset(csrValA, 0, nnzA * sizeof(float));

	if (isSymmetric) {
		for (unsigned long i = 0; i < nnzA_mtx_report; i++)
		{
			if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i]) {
				int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
				csrColIdxA[offset] = csrColIdxA_tmp[i];
				csrValA[offset] = csrValA_tmp[i];
				csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

				offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
				csrColIdxA[offset] = csrRowIdxA_tmp[i];
				csrValA[offset] = csrValA_tmp[i];
				csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
			}
			else {
				int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
				csrColIdxA[offset] = csrColIdxA_tmp[i];
				csrValA[offset] = csrValA_tmp[i];
				csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
			}
		}
	}
	else {
		for (unsigned long i = 0; i < nnzA_mtx_report; i++)
		{
			int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
			csrColIdxA[offset] = csrColIdxA_tmp[i];
			csrValA[offset] = csrValA_tmp[i];
			csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
		}
	}

	// free tmp space
	free(csrColIdxA_tmp);
	free(csrValA_tmp);
	free(csrRowIdxA_tmp);
	free(csrRowPtrA_counter);
	return 0;
}

void cal_shape_size(int M, int N, int mF, int x1, int y1, int a1, int b1, int &x, int &y, int &a, int &b, float &gain)
{
	if (M > mF) {
		gain = (float)M/(float)(mF);
		x = (int)(x1 * gain);
		y = (int)(y1 * gain);
		a = (int)(a1 * gain);
		b = (int)(b1 * gain);
	}
	else if (M < mF) {
		gain = (float)mF/(float)M;
		x = (int)(x1 / gain);
		y = (int)(y1 / gain); 
		a = (int)(a1 / gain);
		b = (int)(b1 / gain);
	}
	else { // M == mF
		gain = 1.0; 
		x = x1;
		y = y1; 
		a = a1;
		b = b1;
	}
	if (x < 0) x = 0; if (x > M-1) x = M-1; 
	if (y < 0) y = 0; if (y > N-1) y = N-1;
	if (y + a > N) a = N - y; 
	if (x + b > M) b = M - x;
    if (a > b) 
        a = b;
    else 
        b = a;
}

void partition_shapes(char *matrix_path, int M, int N, int nnz, int* &csrRow, 
            int* &csrCol, float* &csrVal, TaiChi_new *taichi_new)
{
    string path(matrix_path);
    //category is diagnal
    string category;
    Point_t size{};
    vector<Point_t> shapePoints;
    int mF = 0, num_shapes = 0;
    vector<double> widths;
#if TIMING
    gpu_timer getSdf_timer;
    getSdf_timer.start();
#endif
    getSdf(path, M, N, nnz, csrRow, csrCol, csrVal,category, size, num_shapes, shapePoints, widths);
#if TIMING
    float getSdf_time = getSdf_timer.stop();
#endif

    mF = size.x;
    shape *h = (shape *)malloc(sizeof(shape) * num_shapes);
    TaiChi *taichi = (TaiChi*)malloc(sizeof(TaiChi) * num_shapes);
    memset(h, 0, sizeof(shape) * num_shapes);
    memset(taichi, 0, sizeof(TaiChi) * num_shapes);

    int dia_shapes = 0, num_negs = 0, dense_nnz = 0, sparse_nnz = 0, zero_elements = 0;

    // record nnz density of diagonal blocks and ultra-sparse matrix
#ifdef RECORD_NNZ_DENSITY    
    FILE * f = fopen("nnz_density.txt", "a+");
    if (f != NULL) {
        char ch=fgetc(f);
        if (ch == EOF) {// file is empty 
            fprintf(f, "Matrix M N nnz dia_density nDense nSparse nZero\n");
        }
    }
    else {
        printf("open file failed\n");
    }
    fprintf(f, "%s %d %d %d ", matrix_path, M, N, nnz);
    nnz_density = 0.0;
#endif

#if TIMING
    cpu_timer partition_timer;
    partition_timer.start();
#endif
#pragma omp parallel for default(shared) num_threads(4)
	for(int i = 0; i<num_shapes; i++){
#if DEBUG
		printf("\nBegin %d-th shape: ", i);
#endif
		h[i].id = i;
        strcpy(h[i].category, category.c_str());
        int i2 = i << 1;
        h[i].x1 = shapePoints[i2].x;
        h[i].y1 = shapePoints[i2].y;
        h[i].a1 = shapePoints[i2 + 1].x - h[i].x1;
        h[i].b1 = shapePoints[i2 + 1].y - h[i].y1;
        float gain;
        cal_shape_size(M, N, mF, h[i].x1, h[i].y1, h[i].a1, h[i].b1, h[i].x, h[i].y, h[i].a, h[i].b, gain);
#if DEBUG
        printf("x1:%d,y1:%d, a1:%d, b1:%d\n", h[i].x1, h[i].y1, h[i].a1, h[i].b1);
	    printf("xStart:%d, yStart:%d, N_sub:%d, M_sub:%d\n", h[i].x, h[i].y, h[i].a, h[i].b);
#endif
		if(strcmp(h[i].category, "diagonal") == 0) {
            if (h[i].a < 1000 ||  h[i].b < 1000) {
                printf("too small shape, let's skip it!\n");
                taichi[i].neg = 0;
                taichi[i].maxNumZero = 0;
                taichi[i].shape = 0; // 0 means diagonal
                taichi[i].nnz = 0;
                taichi[i].width = 0;
                taichi[i].M_sub = M;
                taichi[i].N_sub = N;
                taichi[i].xStart = 0;
                taichi[i].yStart = 0;
                printf("End %d-th shape.\n", i);
                continue;
            }
			partition_dia_shape(M, N, nnz, csrRow, csrCol, csrVal, gain, h[i], taichi[i]);
            if (taichi[i].nnz > 0) {
                dia_shapes++; 
                num_negs += taichi[i].neg;
                dense_nnz += taichi[i].nnz;
            }
		// end diagonal
		else {
			// rectangle and triangle
			taichi[i].neg = 0;
			taichi[i].maxNumZero = 0;
			taichi[i].shape = 2; // 2 means rectangle or triangle
			taichi[i].nnz = 0;
			taichi[i].width = 0;
			taichi[i].M_sub = M;
			taichi[i].N_sub = N;
			taichi[i].xStart = 0;
			taichi[i].yStart = 0;

			h[i].b = 0;
			h[i].a = 0;
			h[i].x = 0;
			h[i].y = 0;
		}// end rectangle and triangle
        }
#if DEBUG
        printf("End %2d-th shape.\n", i);
#endif
    }

    // filtering empty shapes and merge all non-empty diagonal shapes
#if DEBUG
    printf("filtering empty shapes and merge all non-empty diagonal shapes.\n");
#endif
    int shape_index[dia_shapes] = {0};
    int shape_index_len = 0;
    if (dia_shapes > 0) {
        taichi_new->xStart = (int *)malloc(sizeof(int) * dia_shapes);
        taichi_new->yStart = (int *)malloc(sizeof(int) * dia_shapes);
        taichi_new->xStop = (int *)malloc(sizeof(int) * dia_shapes);
        taichi_new->nDiasPtr = (int *)malloc(sizeof(int) * (dia_shapes + 1));
        taichi_new->neg_offsets = (int *)malloc(sizeof(int) * num_negs);

        taichi_new->nDiasPtr[0] = 0;
        int d = 0, tmp_num_neg = 0;
        for (int i = 0; i < num_shapes; i++)
        {
            if (taichi[i].nnz > 0) {
                shape_index[shape_index_len++] = i;
                taichi_new->xStart[d] = taichi[i].xStart;
                taichi_new->yStart[d] = taichi[i].yStart;
                taichi_new->xStop[d] = taichi[i].xStart + taichi[i].M_sub;
                taichi_new->nDiasPtr[d+1] = taichi_new->nDiasPtr[d] + taichi[i].neg;
                memcpy(&taichi_new->neg_offsets[tmp_num_neg], taichi[i].neg_offsets, taichi[i].neg*sizeof(int));
                d++;
                tmp_num_neg += taichi[i].neg;

                for (int n = 0; n < taichi[i].neg; n++)
                    zero_elements += taichi[i].numZeroNeg[n];
            }
        }
        if (d != dia_shapes) printf("Error: filtering empty shapes, d=%d, dia_shapes=%d\n", d, dia_shapes);
        if (shape_index_len != dia_shapes)
            printf("Error: filtering empty shapes, shape_index_len=%d, dia_shapes=%d\n",
                shape_index_len, dia_shapes);
        taichi_new->row_start = min(taichi_new->xStart, dia_shapes);
        taichi_new->row_stop = max(taichi_new->xStop, dia_shapes);
    }
#if TIMING
    float partition_time = partition_timer.stop();
#endif

    sparse_nnz = nnz - dense_nnz;
    printf("dia_shapes=%d, num_negs=%d, dense_nnz=%d, sparse_nnz=%d\n", 
            dia_shapes, num_negs, dense_nnz, sparse_nnz);

#ifdef RECORD_NNZ_DENSITY   
    if (num_negs > 0)        
        fprintf(f, "%f ", nnz_density/(float)num_negs);
    else
        fprintf(f, "0 ");
    fprintf(f, "%d %d %d\n", dense_nnz, sparse_nnz, zero_elements);
    fclose(f);
#endif
    taichi_new->dia_shapes = dia_shapes;
    taichi_new->dense_nnz = dense_nnz;
    taichi_new->sparse_nnz = sparse_nnz;
    taichi_new->zero_elements = zero_elements;
    taichi_new->num_negs = num_negs;
#if DEBUG
    printf("row_start=%d, row_stop=%d\n", taichi_new->row_start, taichi_new->row_stop);
    printf("dense_nnz=%d zero_elements=%d sparse_nnz=%d\n", dense_nnz, zero_elements, sparse_nnz);

    //processing ultra-sparse shape
	printf("processing ultra-sparse shape\n");
#endif
#if TIMING
    cpu_timer process_ultraSparse_timer;
    process_ultraSparse_timer.start();
#endif
    int *rowPtr_zero = NULL, *colIdx_zero = NULL, 
        *rowPtr_sparse = NULL, *colIdx_sparse = NULL;
    if (zero_elements > 0) {
        rowPtr_zero = (int *)malloc(sizeof(int) * (M+1));
        colIdx_zero = (int *)malloc(sizeof(int) * zero_elements);
        memset(rowPtr_zero, 0, sizeof(int) * (M+1));
        memset(colIdx_zero, 0, sizeof(int) * zero_elements);

        // get zero elements in dense diagonals
        int nnz_index = 0;
        int *row_indices = (int *)malloc(sizeof(int) * zero_elements);
        int *col_indices = (int *)malloc(sizeof(int) * zero_elements);
        memset(row_indices, 0, sizeof(int) * zero_elements);
        memset(col_indices, 0, sizeof(int) * zero_elements);
        // get row and column indices of zero elements
        for (int j = 0; j < dia_shapes; j++)
        {
            int id = shape_index[j];
            for (int d = 0; d < taichi[id].neg; d++)
            {
                for (int k = 0; k < taichi[id].numZeroNeg[d]; k++)
                {
                    row_indices[nnz_index] = taichi[id].xStart + taichi[id].rowZeroNeg[d][k];
                    rowPtr_zero[row_indices[nnz_index]]++;
                    int column_index = taichi[id].yStart + taichi[id].rowZeroNeg[d][k] + 
                                       taichi[id].neg_offsets[d];
                    col_indices[nnz_index] = column_index << 1;
                    nnz_index ++;
                }
            }
        }
        if (nnz_index != zero_elements) 
            printf("ERROR: get wrong zero elements, nnz_index=%d, zero_elements=%d\n", 
                    nnz_index, zero_elements);
        int old_val = rowPtr_zero[0], new_val = 0;
        rowPtr_zero[0] = 0;
        for (int i = 1; i <= M; i++)
        {
            new_val = rowPtr_zero[i];
            rowPtr_zero[i] = old_val + rowPtr_zero[i-1];
            old_val = new_val;
        }
        if (rowPtr_zero[M] != zero_elements) 
            printf("ERROR: convertion of row_indices to rowPtr_zero is wrong!\n");
        int *rowPtr_tmp = (int *)malloc(sizeof(int) * M);
        memset(rowPtr_tmp, 0, sizeof(int) * M);

        for (int i = 0; i < zero_elements; i++)
        {
            int offset = rowPtr_zero[row_indices[i]] + rowPtr_tmp[row_indices[i]];
            colIdx_zero[offset] = col_indices[i];
            rowPtr_tmp[row_indices[i]]++;
        }
        free(row_indices); free(col_indices); free(rowPtr_tmp);
    }
    if (sparse_nnz > 0 && dense_nnz > 0) {
        rowPtr_sparse = (int *)malloc(sizeof(int) * (M+1));
        colIdx_sparse = (int *)malloc(sizeof(int) * sparse_nnz);
        memset(rowPtr_sparse, 0, sizeof(int) * (M+1));
        memset(colIdx_sparse, 0, sizeof(int) * sparse_nnz);

        // get sparse non-zero elements
        rowPtr_sparse[0] = 0;
        int nnz_index = 0;
        for (int r = 0; r < M; r++) {
            rowPtr_sparse[r+1] = rowPtr_sparse[r];
            for (int j = csrRow[r]; j < csrRow[r+1]; j++) {
                if ( fabs(csrVal[j]-FLT_MAX) > ZERO ) {
                    rowPtr_sparse[r+1]++;
                    colIdx_sparse[nnz_index++] = (csrCol[j]<<1) + 1;
                }
            }
        }
        if (nnz_index != sparse_nnz) printf("ERROR: get wrong sparse non-zero elements\n");
    }

    // merging two parts
#if DEBUG
    printf("merging two parts\n");
#endif
    int nnz_shape = zero_elements + sparse_nnz;
    int nnz_index = 0, zero_index = 0, sparse_index = 0;
    if (dense_nnz == 0) {
        taichi_new->csrRow = csrRow;
        taichi_new->csrCol = csrCol;
    }
    else if (dense_nnz > 0 && nnz_shape > 0) {
        taichi_new->csrRow = (int *)malloc(sizeof(int) * (M+1));
        taichi_new->csrCol = (int *)malloc(sizeof(int) * nnz_shape);
        memset(taichi_new->csrRow, 0, sizeof(int) * (M+1));
        memset(taichi_new->csrCol, 0, sizeof(int) * nnz_shape);
        if (zero_elements > 0 && sparse_nnz > 0) {
            for (int row = 0; row < M; row ++)
            {
                int num_zero_tmp = rowPtr_zero[row+1] - rowPtr_zero[row];
                int num_sparse_tmp = rowPtr_sparse[row+1] - rowPtr_sparse[row];
                taichi_new->csrRow[row+1] = rowPtr_zero[row+1] + rowPtr_sparse[row+1];
                memcpy(&taichi_new->csrCol[nnz_index], &colIdx_zero[zero_index], sizeof(int) * num_zero_tmp);
                nnz_index += num_zero_tmp;
                memcpy(&taichi_new->csrCol[nnz_index], &colIdx_sparse[sparse_index], sizeof(int) * num_sparse_tmp);
                nnz_index += num_sparse_tmp;
                zero_index += num_zero_tmp;
                sparse_index += num_sparse_tmp;
            }
            if (nnz_index != nnz_shape) 
                printf("ERROR: wrong merging, nnz_index=%d, nnz_shape=%d\n", nnz_index, nnz_shape);
            if (zero_index != zero_elements) 
                printf("ERROR: wrong merging, zero_index=%d, zero_elements=%d\n", zero_index, zero_elements);
            if (sparse_index != sparse_nnz) 
                printf("ERROR: wrong merging, sparse_index=%d, sparse_nnz=%d\n", sparse_index, sparse_nnz);
        }
        else if (zero_elements == 0) {
            memcpy(taichi_new->csrRow, rowPtr_sparse, sizeof(int) * (M+1));
            memcpy(taichi_new->csrCol, colIdx_sparse, sizeof(int) * sparse_nnz);
        }
        else { // sparse_nnz == 0 && zero_elements > 0 
            memcpy(taichi_new->csrRow, rowPtr_zero, sizeof(int) * (M+1));
            memcpy(taichi_new->csrCol, colIdx_zero, sizeof(int) * zero_elements);
        }
    }
#if DEBUG
    printf("End processing ultra-sparse shape\n");
#endif
#if TIMING
    float process_ultraSparse_time = process_ultraSparse_timer.stop();
    printf("getSdf:%.4f partition_dense:%.4f partition_sparse:%.4f ms\n", getSdf_time, partition_time, process_ultraSparse_time);
#endif

    // free space
    if (rowPtr_zero != NULL)    
        free(rowPtr_zero); 
    if (colIdx_zero != NULL)
        free(colIdx_zero);
    if (rowPtr_sparse != NULL)
        free(rowPtr_sparse); 
    if (colIdx_sparse != NULL)  
        free(colIdx_sparse);

    free_taichi(num_shapes, taichi);
    
    if (h != NULL)
    {
        free(h);
        h = NULL;
    }
}

void free_taichi(int num_shapes, TaiChi *taichi)
{
    for (int i = 0; i < num_shapes; i++)
    {
        if (taichi[i].neg_offsets != NULL) {
            free(taichi[i].neg_offsets);
            taichi[i].neg_offsets = NULL;
        }
        if (taichi[i].numZeroNeg != NULL)
        {
            free(taichi[i].numZeroNeg);
            taichi[i].numZeroNeg = NULL;
        }
        for (int d = 0; d < taichi[i].neg; d++)
        {
            if (taichi[i].rowZeroNeg != NULL)
            {
                if (taichi[i].rowZeroNeg[d] != NULL)
                {
                    free(taichi[i].rowZeroNeg[d]);
                    taichi[i].rowZeroNeg[d] = NULL;
                }
            }
        }
        if (taichi[i].rowZeroNeg != NULL)
        {
            free(taichi[i].rowZeroNeg);
            taichi[i].rowZeroNeg = NULL;
        }
    }
    if (taichi != NULL)
    {
        free(taichi);
        taichi = NULL;
    }
}

void partition_dia_shape(int M, int N, int nnz, int* &csrRow, int* &csrCol, float* &csrVal, 
            float gain, shape &hi, TaiChi &taichi_i)
{
	taichi_i.M_sub = hi.b;
	taichi_i.N_sub = hi.a;
	taichi_i.xStart = hi.x;
	taichi_i.yStart = hi.y;
	int xStart = taichi_i.xStart, yStart = taichi_i.yStart;
	int M_sub = taichi_i.M_sub, N_sub = taichi_i.N_sub;

	int width;
	// set the width of diagonal strip
	int smaller_dim = (M_sub > N_sub)? N_sub: M_sub;
	if (smaller_dim>= 10000000)
		width = smaller_dim * 0.000001;
	else if (smaller_dim >= 1000000)
		width = smaller_dim * 0.00001;
	else if (smaller_dim >= 100000)
		width = smaller_dim * 0.001;
	else if (smaller_dim >= 10000)
		width = smaller_dim * 0.01;
	else 
		width = smaller_dim * 0.1;

	/* this is the theoretical value in pure mathematics, which is too small for actual use
	int width = h[i].a - sqrt(h[i].a*h[i].a-h[i].area1*gain) + 2;
	int width = h[i].a - sqrt(h[i].a*h[i].a-h[i].area1*gain)+20 ;
	*/

	taichi_i.width = width;
	taichi_i.shape = 0; // 0 means diagonal
	int num_diagonals = 2*width-1;
#if DEBUG
    printf("width=%d num_diagonals=%d smaller_dim=%d\n", width, num_diagonals, smaller_dim);
#endif
	
	int *rowID_nonZero_each_dia = (int *)malloc(sizeof(int) * num_diagonals*smaller_dim);
    int (*rowID_nonZero_each_dia2)[smaller_dim] = (int(*)[smaller_dim])rowID_nonZero_each_dia; // store the row index of non-zero for each diagonal

	int *nnz_each_dia = NULL;
	nnz_each_dia = (int*)malloc(sizeof(int)*num_diagonals);
	memset(nnz_each_dia, 0, sizeof(int)*num_diagonals);

	int n_diagonal = 0;
	int diagonal_offset = hi.y - hi.x;
	int r = 0, c = 0, nnz_shape = 0;
	for (int i = 0; i < M; i++) {
		for (int j = csrRow[i]; j < csrRow[i+1]; j++) {
			r = i; c = csrCol[j];
			n_diagonal = c - r - (yStart - xStart) + width - 1;
			int new_r = r + diagonal_offset;
			if(	r >= hi.x && r < (hi.x+hi.b) && c >= hi.y && c < (hi.y+hi.a)
				&& c >= (new_r-width+1) && c <= (new_r+width-1) 
				&& fabs(csrVal[j]-FLT_MAX)>ZERO ){
				rowID_nonZero_each_dia2[n_diagonal][nnz_each_dia[n_diagonal]++] = r - hi.x;
			}	
		}// end for j
	} // end for i

	int *neg_offsets = NULL;
	// save the offset of each dense diagonal
	neg_offsets = (int*)malloc(sizeof(int) * num_diagonals);
	memset(neg_offsets, 0, sizeof(int) * num_diagonals);
	int *start  = NULL, *end = NULL; // save the start and end row index of each dense diagonal
	start = (int*)malloc(sizeof(int) * num_diagonals);
	end   = (int*)malloc(sizeof(int) * num_diagonals);
	memset(start, 0, sizeof(int) * num_diagonals);
	memset(end,   0, sizeof(int) * num_diagonals);

	int neg = 0;
	int dia_offsets_temp = 0;
	float density_temp = 0.0;
	int start_temp = 0, end_temp = 0;

	// Find dense diagonals;
	for(int n = 0; n < num_diagonals; n++)
	{
		dia_offsets_temp = n - width + 1;
		cal_start_end(M_sub, N_sub, width, dia_offsets_temp, n, start_temp, end_temp);
		density_temp = float(nnz_each_dia[n]) / float(end_temp - start_temp);
		if (density_temp >= 0.6) {
			start[neg] = start_temp;
			end[neg] = end_temp;
			neg_offsets[neg] = dia_offsets_temp;
			neg ++;
#ifdef RECORD_NNZ_DENSITY 
            nnz_density += density_temp;
#endif
		}
	}

	taichi_i.neg = neg;
	// process diagonals with non-zero density greater than threshold
	if (neg > 0) {
		taichi_i.neg_offsets = (int *)malloc(sizeof(int)*neg);
		memcpy(taichi_i.neg_offsets, neg_offsets, neg * sizeof(int));
		taichi_i.start = (int*)malloc(sizeof(int)*neg);
		memcpy(taichi_i.start, start, neg * sizeof(int));
		taichi_i.end = (int*)malloc(sizeof(int)*neg);
		memcpy(taichi_i.end, end, neg * sizeof(int));

		int maxNumZero=0; // record the maximal number of zeros in one dense diagonal
		// store the number of zeros for each dense diagonal
		taichi_i.numZeroNeg = (int *)malloc(sizeof(int)*neg);
		taichi_i.cStart = (int *)malloc(sizeof(int)*neg);

		for(int d = 0; d < neg; d++)
		{
			n_diagonal = taichi_i.neg_offsets[d] + width - 1;
			nnz_shape += nnz_each_dia[n_diagonal];
			taichi_i.numZeroNeg[d] = taichi_i.end[d] - taichi_i.start[d] - nnz_each_dia[n_diagonal];
			maxNumZero = (taichi_i.numZeroNeg[d] > maxNumZero)? taichi_i.numZeroNeg[d]:maxNumZero;
			if(taichi_i.neg_offsets[d] > 0)
				taichi_i.cStart[d] = taichi_i.neg_offsets[d];
			else
				taichi_i.cStart[d] = 0;
			
		}
		taichi_i.nnz = nnz_shape; 
		taichi_i.maxNumZero = maxNumZero;

		// rowZeroNeg record the row index of each zero
		taichi_i.rowZeroNeg = (int **)malloc(sizeof(int*)*neg);
		memset(taichi_i.rowZeroNeg, 0, sizeof(int*)*neg);
		if(maxNumZero > 0) {
			for (int k = 0; k < neg; k++)
			{
				taichi_i.rowZeroNeg[k] = NULL;
				taichi_i.rowZeroNeg[k] = (int *) malloc(sizeof(int) * maxNumZero);
			}
		}

		for(int d = 0; d < neg; d++)
		{
			int rNnzIndex = 0;
			int n_diagonal = neg_offsets[d] + width -1;
			int rZeroIndex = 0;
			for (int r = taichi_i.start[d]; r < taichi_i.end[d]; r++)
			{
				if (r < rowID_nonZero_each_dia2[n_diagonal][rNnzIndex]) {
					taichi_i.rowZeroNeg[d][rZeroIndex++] = r;
				}
				else {
					if(rNnzIndex < nnz_each_dia[n_diagonal])
						rNnzIndex ++;
					else
						taichi_i.rowZeroNeg[d][rZeroIndex++] = r;
				}
					
			}
		}
	} // end if neg>0

	if(rowID_nonZero_each_dia != NULL) {
		free(rowID_nonZero_each_dia);
		rowID_nonZero_each_dia = NULL;
	}
	if(nnz_each_dia != NULL) {
        free(nnz_each_dia); 
        nnz_each_dia = NULL;
    }
	if(start != NULL) {
        free(start); 
        start = NULL;
    }
	if(end != NULL) {
        free(end); 
        end = NULL;
    }

	// mark non-zero within the dense diagonals
	if (neg > 0) {
		nnz_shape = 0;
		for (int i = 0; i < M; i++) 
        {
			for (int j = csrRow[i]; j < csrRow[i+1]; j++) 
            {
				r = i;
				c = csrCol[j];
				int new_r = r + diagonal_offset;
				if(	r >= hi.x && r < (hi.x+hi.b) && c >= hi.y && c < (hi.y+hi.a)
					&& c >= (new_r-width+1) && c <= (new_r+width-1) 
					&& fabs(csrVal[j]-FLT_MAX)>ZERO ) {
					int r_sub = r - hi.x;
					int c_sub = c - hi.y;
					for(int d = 0; d <neg; d++)
					{	
						if(c_sub - r_sub == neg_offsets[d]) {					
							nnz_shape++;
							csrVal[j] = FLT_MAX;// mark the partitioned non-zeros
						}
					}
				}
			}
		}
        if (nnz_shape != taichi_i.nnz) 
            printf("This shape's PARTITION IS WRONG!!\n");
	}
	if(neg_offsets != NULL) {
        free(neg_offsets); 
        neg_offsets = NULL;
    }
}

void cal_start_end (int M_sub, int N_sub, int width, int dia_offsets_temp, 
            int n, int &start_temp, int &end_temp)
{
	if(M_sub >= N_sub) {
		if(n >= width -1) {
			start_temp = 0;
			end_temp = N_sub - dia_offsets_temp;
		}
		else {
			if(abs(dia_offsets_temp) > (M_sub - N_sub)) {
				start_temp = abs(dia_offsets_temp);
				end_temp = M_sub;
			}
			else {
				start_temp = abs(dia_offsets_temp);
				end_temp = N_sub + abs(dia_offsets_temp);
			}
		}
	}
	else if(M_sub < N_sub) {
		if(n <= width - 1) {
			start_temp = abs(dia_offsets_temp);
			end_temp = M_sub;
		}
		else {
			if(abs(dia_offsets_temp) > (N_sub - M_sub)) {
				start_temp = 0;
				end_temp = N_sub - abs(dia_offsets_temp);
			}
			else {
				start_temp = 0;
				end_temp = M_sub;
			}
		}
	}
}

void queryDevice()  
{  
    cudaDeviceProp deviceProp;  
    int deviceCount = 0;  
    cudaError_t cudaError;  
    cudaError = cudaGetDeviceCount(&deviceCount);  
	cout<<"cudaError = "<<cudaError<<endl;
    for (int i = 0; i < deviceCount; i++)  
    {  
        cudaError = cudaGetDeviceProperties(&deviceProp, i);  
        cout << "Device " << i << "'s main property: " << endl;  
        cout << "Device Name: " << deviceProp.name << endl;  
        cout << "Global Memory of device: " << deviceProp.totalGlobalMem / 1024 / 1024 << " MB" << endl;  
        cout << "Maximal available shared memory for a block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << endl;  
        cout << "Number of available registers for a block: " << deviceProp.regsPerBlock << endl;  
        cout << "Maximal number of threads for a block: " << deviceProp.maxThreadsPerBlock << endl;  
        cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << endl;  
        cout << "Number of multi processors: " << deviceProp.multiProcessorCount << endl;  
    }  
	cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)  printf("cudaSetDevice failed!");

	int device = -1;
	cudaStatus = cudaGetDevice(&device);
    if (cudaStatus != cudaSuccess)  printf("cudaGetDevice failed!");
	cout<<"\nThe device now beening used is device "<<device<<endl<<endl;
}   

int max(int *a, int len)
{
	int max = a[0];
	for(int i = 1; i < len; i ++)
		if(a[i] > max)
			max = a[i];
	return max;
}

int min(int *a, int len)
{
	int min = a[0];
	for(int i = 1; i < len; i ++)
		if(a[i] < min)
			min = a[i];
	return min;
}

// if (dia_shapes == 1)
template <  unsigned int THREADS_PER_VECTOR,
            unsigned int VECTORS_PER_BLOCK>
__global__ void CSR_coop_spmv_kernel_hybrid1(
    const int rows,
    const int *Ap,
    const int *Aj,
    int row_start_dia, int row_stop_dia, int dia_shapes, int num_negs,
    const int *data, const int *neg_offsets, const float *x, float *y)
{
    __shared__ volatile float sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR/2];  // padded to avoid reduction conditionals
    __shared__ volatile int ptrs[VECTORS_PER_BLOCK][2];
    extern __shared__ int s_taichi[];
    int *s_diaShps = s_taichi, *s_offsets = &s_taichi[5*dia_shapes];

    const int THREADS_PER_BLOCK = THREADS_PER_VECTOR * VECTORS_PER_BLOCK;
    const int thread_id   = blockDim.x * blockIdx.x + threadIdx.x;    // global thread index
    const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const int vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const int vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const int num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for (int i = threadIdx.x; i < 5*dia_shapes; i += THREADS_PER_BLOCK)
        s_diaShps[i] = data[i];
    for (int i = threadIdx.x; i < num_negs; i += THREADS_PER_BLOCK)
        s_offsets[i] = neg_offsets[i];

    __syncthreads();

    for(int row = vector_id; row < rows; row += num_vectors)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        if(thread_lane < 2)
            ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];

        __syncwarp();   // jusr for Tesla V100
        const int row_start    = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const int row_end      = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];

        float sum = 0.0;
            
        // SpMV for ultra-sparse sub-matrix
        for(int jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) 
        {
            int curColIdx = Aj[jj];
            int sign = (curColIdx & 1);
            int colIdx = (curColIdx >> 1 );
            FUNC2
        }
        // SpMV for diagonal blocks
        if (row >= row_start_dia && row < row_stop_dia) {
            int yStart_d = s_diaShps[2];
            int common = row + yStart_d - s_diaShps[0];
            int stop_cSub = yStart_d + row_stop_dia - row_start_dia;
            for (int n = s_diaShps[3] + thread_lane; n < s_diaShps[4]; n += THREADS_PER_VECTOR)
            {
                int col = common + s_offsets[n];
                if (col >= yStart_d && col < stop_cSub) 
                    sum += x[col];
            }
        }

        sdata[threadIdx.x] = sum;
        float temp;

        for (int stride = THREADS_PER_VECTOR/2; stride>0; stride/=2)
        {
            __syncwarp(); // jusr for Tesla V100
            temp = sdata[threadIdx.x + stride];
            sdata[threadIdx.x] = sum = sum + temp;
        }

        __syncwarp(); // jusr for Tesla V100
        if (thread_lane == 0) {
            y[row] = sdata[threadIdx.x];
        }
    }
}

// if (dia_shapes > 1 && num_negs/dia_shapes == 1), parallel for diagonal shapes
template <  unsigned int THREADS_PER_VECTOR,
            unsigned int VECTORS_PER_BLOCK>
__global__ void CSR_coop_spmv_kernel_hybrid2(
    const int rows,
    const int *Ap,
    const int *Aj,
    int row_start_dia, int row_stop_dia, int dia_shapes, int num_negs,
    const int *data, const int *neg_offsets, const float *x, float *y)
{
    __shared__ volatile float sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR/2];  // padded to avoid reduction conditionals
    __shared__ volatile int ptrs[VECTORS_PER_BLOCK][2];
    extern __shared__ int s_taichi[];
    int *s_diaShps = s_taichi, *s_offsets = &s_taichi[5*dia_shapes];

    const int THREADS_PER_BLOCK = THREADS_PER_VECTOR * VECTORS_PER_BLOCK;
    const int thread_id   = blockDim.x * blockIdx.x + threadIdx.x;    // global thread index
    const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const int vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const int vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const int num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for (int i = threadIdx.x; i < 5*dia_shapes; i += THREADS_PER_BLOCK)
        s_diaShps[i] = data[i];
    for (int i = threadIdx.x; i < num_negs; i += THREADS_PER_BLOCK)
        s_offsets[i] = neg_offsets[i];

    __syncthreads();

    for(int row = vector_id; row < rows; row += num_vectors)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        if(thread_lane < 2)
            ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];

        __syncwarp();   // jusr for Tesla V100
        const int row_start    = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const int row_end      = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];

        float sum = 0.0;
            
        // SpMV for ultra-sparse sub-matrix
        for(int jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) 
        {
            int curColIdx = Aj[jj];
            int sign = (curColIdx & 1);
            int colIdx = (curColIdx >> 1 );
            FUNC2
        }
        // SpMV for diagonal block
        if (row >= row_start_dia && row < row_stop_dia) {
            for (int d = thread_lane; d < dia_shapes; d += THREADS_PER_VECTOR)
            {
                int index = 5 * d;
                int xStart_d = s_diaShps[index];
                int xStop_d = s_diaShps[index+1];
                if (row >= xStart_d && row < xStop_d) {
                    int yStart_d = s_diaShps[index + 2];
                    int common = row + yStart_d - xStart_d;
                    int stop_cSub = yStart_d + xStop_d - xStart_d;
                    for (int n = s_diaShps[index + 3]; n < s_diaShps[index + 4]; n++)
                    {
                        int col = common + s_offsets[n];
                        if (col >= yStart_d && col < stop_cSub)
                            sum += x[col];
                    }
                }
            }
        }

        sdata[threadIdx.x] = sum;
        float temp;

        for (int stride = THREADS_PER_VECTOR/2; stride>0; stride/=2)
        {
            __syncwarp();   // jusr for Tesla V100
            temp = sdata[threadIdx.x + stride];
            sdata[threadIdx.x] = sum = sum + temp;
        }

        __syncwarp();   // jusr for Tesla V100
        if (thread_lane == 0)  {
            y[row] = sdata[threadIdx.x];
        }
    }
}

// if (dia_shapes > 1 && num_negs/dia_shapes > 1), parallel for diagonals
template <  unsigned int THREADS_PER_VECTOR,
            unsigned int VECTORS_PER_BLOCK>
__global__ void CSR_coop_spmv_kernel_hybrid3(
    const int rows,
    const int *Ap,
    const int *Aj,
    int row_start_dia, int row_stop_dia, int dia_shapes, int num_negs,
    const int *data, const int *neg_offsets, const float *x, float *y)
{
    __shared__ volatile float sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR/2];  // padded to avoid reduction conditionals
    __shared__ volatile int ptrs[VECTORS_PER_BLOCK][2];
    extern __shared__ int s_taichi[];
    int *s_diaShps = s_taichi, *s_offsets = &s_taichi[5*dia_shapes];

    const int THREADS_PER_BLOCK = THREADS_PER_VECTOR * VECTORS_PER_BLOCK;
    const int thread_id   = blockDim.x * blockIdx.x + threadIdx.x;    // global thread index
    const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const int vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const int vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const int num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for (int i = threadIdx.x; i < 5*dia_shapes; i += THREADS_PER_BLOCK)
        s_diaShps[i] = data[i];
    for (int i = threadIdx.x; i < num_negs; i += THREADS_PER_BLOCK)
        s_offsets[i] = neg_offsets[i];

    __syncthreads();

    for(int row = vector_id; row < rows; row += num_vectors)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        if(thread_lane < 2)
                ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];

        __syncwarp();   // jusr for Tesla V100
        const int row_start    = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const int row_end      = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];

        float sum = 0.0;
            
        // accumulate local sums
        for(int jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) 
        {
            int curColIdx = Aj[jj];
            int sign = (curColIdx & 1);
            int colIdx = (curColIdx >> 1 );
            FUNC2
        }

        if (row >= row_start_dia && row < row_stop_dia) {
            for (int d = 0; d < dia_shapes; d++)
            {
                int index = 5 * d;
                int xStart_d = s_diaShps[index];
                int xStop_d = s_diaShps[index + 1];
                if (row >= xStart_d && row < xStop_d) {
                    int yStart_d = s_diaShps[index + 2];
                    int common = row + yStart_d - xStart_d;
                    int stop_cSub = yStart_d + (xStop_d - xStart_d);
                    for (int n = s_diaShps[index + 3] + thread_lane; n < s_diaShps[index + 4]; n += THREADS_PER_VECTOR)
                    {
                        int col = common + s_offsets[n];
                        if (col >= yStart_d && col < stop_cSub)
                            sum += x[col];
                    }
                }
            }
        }

        sdata[threadIdx.x] = sum;
        float temp;

        for (int stride = THREADS_PER_VECTOR/2; stride>0; stride/=2)
        {
            __syncwarp(); // just for Tesla V100
            temp = sdata[threadIdx.x + stride];
            sdata[threadIdx.x] = sum = sum + temp;
        }
        __syncwarp(); // just for Tesla V100
        if (thread_lane == 0) {
            y[row] = sdata[threadIdx.x];
        }
    }
}

template <  unsigned int THREADS_PER_VECTOR,
            unsigned int VECTORS_PER_BLOCK>
__global__ void CSR_coop_spmv_kernel(
    const int rows,
    const int *Ap,
    const int *Aj, 
    const float *x, float *y)
{
    __shared__ volatile float sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR/2];  // padded to avoid reduction conditionals
    __shared__ volatile int ptrs[VECTORS_PER_BLOCK][2];

    const int THREADS_PER_BLOCK = THREADS_PER_VECTOR * VECTORS_PER_BLOCK;
    const int thread_id   = blockDim.x * blockIdx.x + threadIdx.x;    // global thread index
    const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const int vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const int vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const int num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for(int row = vector_id; row < rows; row += num_vectors)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        if(thread_lane < 2)
            ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];
        __syncwarp();   // just for Tesla V100
        const int row_start    = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const int row_end      = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];
        
        float sum = 0.0;
            
        // accumulate local sums
        for(int jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) 
            sum += x[Aj[jj]];

        sdata[threadIdx.x] = sum;
        float temp;

        for (int stride = THREADS_PER_VECTOR/2; stride>0; stride/=2)
        {
            __syncwarp(); // just for Tesla V100
            temp = sdata[threadIdx.x + stride];
            sdata[threadIdx.x] = sum = sum + temp;
        }

        __syncwarp(); // just for Tesla V100
        if (thread_lane == 0)   {
            y[row] = sdata[threadIdx.x];
        }
    }
}

template <unsigned int THREADS_PER_VECTOR>
void CSR_coop_spmv_prepare1(int M, int *RowPtr, int *ColIdx, 
                            int row_start, int row_stop, int dia_shapes, int num_negs,
                            int *data, int *neg_offsets, float *x, float *y)
{
    const int THREADS_PER_BLOCK = 256;    // THREADS_PER_BLOCK*FOLDGRAIN < SHARED_MEM_SIZE
    const int VECTORS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
    const int dynamic_smem_bytes = (6*dia_shapes+num_negs)*sizeof(int);
	const int MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(CSR_coop_spmv_kernel_hybrid2<THREADS_PER_VECTOR, VECTORS_PER_BLOCK>, THREADS_PER_BLOCK, dynamic_smem_bytes);

	const int REQUIRED_BLOCKS = (M + VECTORS_PER_BLOCK -1)/VECTORS_PER_BLOCK;
	const int NUM_BLOCKS = std::min<int>(MAX_BLOCKS, REQUIRED_BLOCKS);
    if (dia_shapes == 0)
        CSR_coop_spmv_kernel<THREADS_PER_VECTOR, VECTORS_PER_BLOCK><<<NUM_BLOCKS, THREADS_PER_BLOCK, 0>>>(
            M, RowPtr, ColIdx, x, y);
    else if (dia_shapes == 1)
        CSR_coop_spmv_kernel_hybrid1<THREADS_PER_VECTOR, VECTORS_PER_BLOCK><<<NUM_BLOCKS, THREADS_PER_BLOCK, dynamic_smem_bytes>>>(
            M, RowPtr, ColIdx, row_start, row_stop, dia_shapes, num_negs, data, neg_offsets, x, y);
    else if (num_negs/dia_shapes > 1)
        CSR_coop_spmv_kernel_hybrid3<THREADS_PER_VECTOR, VECTORS_PER_BLOCK><<<NUM_BLOCKS, THREADS_PER_BLOCK, dynamic_smem_bytes>>>(
            M, RowPtr, ColIdx, row_start, row_stop, dia_shapes, num_negs, data, neg_offsets, x, y);
    else // (num_negs/dia_shapes == 1)
        CSR_coop_spmv_kernel_hybrid2<THREADS_PER_VECTOR, VECTORS_PER_BLOCK><<<NUM_BLOCKS, THREADS_PER_BLOCK, dynamic_smem_bytes>>>(
            M, RowPtr, ColIdx, row_start, row_stop, dia_shapes, num_negs, data, neg_offsets, x, y);
}

void CSR_coop_spmv_prepare0(int M, int TPV, int *RowPtr, int *ColIdx, 
                            int row_start, int row_stop, int dia_shapes, int num_negs,
                            int *data, int *neg_offsets, float *x, float *y)
{
    if (TPV <= 2)
    {
        CSR_coop_spmv_prepare1<2>(M, RowPtr, ColIdx, 
                                row_start, row_stop, dia_shapes, num_negs,
                                data, neg_offsets, x, y);
        return;
    }
    else if (TPV <= 4)
    {
        CSR_coop_spmv_prepare1<4>(M, RowPtr, ColIdx, 
                                row_start, row_stop, dia_shapes, num_negs,
                                data, neg_offsets, x, y);
        return;
    }
    else if (TPV <= 8)
    {
        CSR_coop_spmv_prepare1<8>(M, RowPtr, ColIdx, 
                                row_start, row_stop, dia_shapes, num_negs,
                                data, neg_offsets, x, y);
        return;
    }
    else if (TPV <= 16)
    {
        CSR_coop_spmv_prepare1<16>(M, RowPtr, ColIdx, 
                            row_start, row_stop, dia_shapes, num_negs,
                            data, neg_offsets, x, y);
        return;
    }

    CSR_coop_spmv_prepare1<32>(M, RowPtr, ColIdx, 
                            row_start, row_stop, dia_shapes, num_negs,
                            data, neg_offsets, x, y);
}

void taichi_SpMV(int M, int N, TaiChi_new *taichi, float *xHostPtr, float *yHostPtr)
{
    float alpha = 1.0;
    int *neg_offsets_device = 0;
    int *d_csrRowPtrA, *d_csrColIdxA;
    int dia_shapes = taichi->dia_shapes;
    int dense_nnz = taichi->dense_nnz;
    int sparse_nnz = taichi->sparse_nnz;
    int zero_elements = taichi->zero_elements;
    int num_negs = taichi->num_negs;
    int row_start = taichi->row_start;
    int row_stop = taichi->row_stop;
    int *data_host = 0, *data_device = 0;

    // prepare data on CPU
    data_host = (int *)malloc(5 * dia_shapes * sizeof(int));
    memset(data_host, 0, 5 * dia_shapes * sizeof(int));
    for (int i = 0; i < dia_shapes; i++) {
        data_host[i * 5 + 0] = taichi->xStart[i];
        data_host[i * 5 + 1] = taichi->xStop[i];
        data_host[i * 5 + 2] = taichi->yStart[i];
        data_host[i * 5 + 3] = taichi->nDiasPtr[i];
        data_host[i * 5 + 4] = taichi->nDiasPtr[i+1];
    }

    // allocate space on GPU
    if (dia_shapes > 0) {
		// malloc space for add operation
        checkcuda(cudaMalloc((void**)&data_device, 5 * dia_shapes * sizeof(int)));
        checkcuda(cudaMalloc((void**)&neg_offsets_device, num_negs * sizeof(int)));
	}
    if((sparse_nnz + zero_elements) > 0) {// there is one ultra sparse matrix
        checkcuda(cudaMalloc((void **)&d_csrRowPtrA, (M+1) * sizeof(int)));
	    checkcuda(cudaMalloc((void **)&d_csrColIdxA, (sparse_nnz + zero_elements) * sizeof(int)));
    }

	// prepare y vector
	float *y_add_device, *y_oth_device, *y_device, *x_device; 
    checkcuda(cudaMalloc((void**)&x_device, N * sizeof(float)));
    checkcuda(cudaMalloc((void**)&y_device, M * sizeof(float)));
    cudaMemset(y_device, M * sizeof(float), 0.0);

    checkcuda(cudaMemcpy(x_device, xHostPtr, N * sizeof(float), cudaMemcpyHostToDevice)); // copy vector x
    // memory copy from host to device
    gpu_timer transfer_timer;
    double all_transfer_time[NUM_TRANSFER];
    for (int i = 0; i < NUM_TRANSFER; i++)
    {
        transfer_timer.start();
        checkcuda(cudaMemcpy(data_device, data_host, 5 * dia_shapes * sizeof(int), cudaMemcpyHostToDevice));
        checkcuda(cudaMemcpy(neg_offsets_device, taichi->neg_offsets, num_negs * sizeof(int), cudaMemcpyHostToDevice));
        checkcuda(cudaMemcpy(d_csrRowPtrA, taichi->csrRow, (M+1) * sizeof(int), cudaMemcpyHostToDevice));
        checkcuda(cudaMemcpy(d_csrColIdxA, taichi->csrCol, (sparse_nnz + zero_elements) * sizeof(int), cudaMemcpyHostToDevice));
        all_transfer_time[i] = transfer_timer.stop();
    }
    
    double transfer_time = average(NUM_TRANSFER, all_transfer_time);
    double var_transfer_time = variance(NUM_TRANSFER, transfer_time, all_transfer_time);

    int TPV = 0;
    if (dia_shapes == 0)
        TPV = sqrt((dense_nnz + sparse_nnz) / M);
    else
        TPV = sqrt((sparse_nnz+zero_elements)/M + num_negs/dia_shapes);
#if DEBUG
    printf("TPV=%d\n", TPV);
#endif

    // warmp up
    for (int i = 0; i < 10; i++)
    {
        CSR_coop_spmv_prepare0(M, TPV, d_csrRowPtrA, d_csrColIdxA, 
            row_start, row_stop, dia_shapes, num_negs, data_device, neg_offsets_device, 
            x_device, y_device);
    }

    // timing kernel
    gpu_timer spmv_timer;
    double all_spmv_time[NUM_RUN];
    for (int i = 0; i < NUM_RUN; i++)
    {
        spmv_timer.start();
        CSR_coop_spmv_prepare0(M, TPV, d_csrRowPtrA, d_csrColIdxA, 
            row_start, row_stop, dia_shapes, num_negs, data_device, neg_offsets_device,
            x_device, y_device);
        all_spmv_time[i] = spmv_timer.stop();
    }
    double spmv_time = average(NUM_RUN, all_spmv_time);
    double var_spmv_time = variance(NUM_RUN, spmv_time, all_spmv_time);
    printf("transfer_time:%.4f spmv_time=%.4f\n", transfer_time, spmv_time);

    checkcuda(cudaMemcpy(yHostPtr, y_device, sizeof(float) * M, cudaMemcpyDeviceToHost));
    

    // FILE *fresult = fopen("taichi_float_V100.txt", "a+");
 	// if (fresult != NULL) {
 	// 	char ch=fgetc(fresult);
 	// 	if (ch == EOF) {// file is empty 
 	// 		fprintf(fresult, "Matrix transfer var_transfer spmv var_spmv\n");
 	// 	}
 	// }
 	// else {
 	// 	printf("open file failed\n");
 	// }
	// fprintf(fresult, "%s %.6f %.6f %.6f %.6f\n", matrixName, transfer_time, var_transfer_time, spmv_time, var_spmv_time);
	// fclose(fresult);
    
	// free resource
    free(data_host);
	checkcuda(cudaFree(x_device));
	checkcuda(cudaFree(y_device));

	if (dia_shapes > 0)
	{
        checkcuda(cudaFree(data_device));
        checkcuda(cudaFree(neg_offsets_device));
	}

    if((sparse_nnz + zero_elements) > 0) // there is one ultra sparse matrix
    {
        checkcuda(cudaFree(d_csrRowPtrA));
	    checkcuda(cudaFree(d_csrColIdxA));
    }
    
	cudaDeviceReset();
}
