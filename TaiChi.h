#include <cuda_runtime.h>
#include "cusparse.h"

// record the basic info of each shape
typedef struct shape{
	int id;
    char category[1024];	// rectangular,triangular, or diagonal
	int x1;					// row index of left upper corner in thumbnail
	int y1;					// column index of left upper corner in thumbnail
	int a1;					// long of the shape in thumbnail
	int b1;					// width of the shape in thumbnail
	int area1;				// number of non-zero pixels of the shape in thumbnail
	int x;					// row index of left upper corner in matrix 
	int y;					// column index of left upper corner in matrix 
	int a;                  // long of the shape in matrix
	int b;                  // width of the shape in matrix
	int width;
}shape;

typedef struct TaiChi {
	char shape; 	//	0-diagnal 1-ultra-sparse 2-triangle-or-rectangle
	int M_sub;		// row length of submatrix
	int N_sub;		// column length of submatrix
	int xStart;		// row offset of submatrix
	int yStart;		// column length of submatrix
	int width;		// diagonal width
		
	// for dense diagnals
	int neg;				// number of dense diagonals
	int maxNumZero;			// maximal number of zeros of dense diagonal 
	int *neg_offsets;		// offset of dense diagonal	
	int *numZeroNeg;		// number of zeros for each dense diagonal
	int **rowZeroNeg;		// row index of zeros in each dense diagonal

	int *start;
	int *end;
	int *cStart;

	// for others
	int nnz;
	int *csrRow;
	int *csrCol;
} TaiChi;

typedef struct TaiChi_new {
    int dia_shapes;         // nDiaShps
    int num_negs;           // nDias
    int dense_nnz;          // nDense
    int zero_elements;      // nZero
    int sparse_nnz;         // nSparse
    int row_start;          // r_start_dia
    int row_stop;           // r_stop_dia
	int *xStart;		    //xStart
	int *yStart;		    // yStart
    int *xStop;             // xStop
	int *nDiasPtr;          // nDiasPtr
	int *neg_offsets;		// offsets
	int *csrRow;            // csRow
	int *csrCol;            // csrCol
}TaiChi_new;

#ifndef VALUE_TYPE
#define VALUE_TYPE float
#endif

// number of spmv runs, just for timing
#ifndef NUM_RUN
#define NUM_RUN 500
#endif

// number of data transfer, just for timing
#ifndef NUM_TRANSFER
#define NUM_TRANSFER 50
#endif

#define ZERO 1e-8

// read a matrix from file, and the matrix is stored with CSR format
int readMtx(char *filename, int &m, int &n, unsigned long &nnzA, 
            int *&csrRowPtrA, int *&csrColIdxA, float *&csrValA);

// MMSparse partition
void partition_shapes(char *sdf_name, int M, int N, int nnz, 
            int* &csRow, int* &csrCol, float* &csrVal, TaiChi_new *neg_format);

// calculate the size of diagonal blocks
void cal_shape_size(int M, int N, int mF, int x1, int y1, int a1, int b1, 
            int &x, int &y, int &a, int &b, float &gain);
         
// partition each diaognal shape
void partition_dia_shape(int M, int N, int nnz, int* &csRow, int* &csrCol, float* &csrVal, 
            float gain, shape &hi, TaiChi &neg_format_i);

// taichi-based SpMV
void taichi_SpMV(int M, int N, TaiChi_new *neg_format, float *xHostPtr, float *yHostPtr);

void queryDevice();

inline void checkcuda(cudaError_t result);
inline void checkcusparse(cusparseStatus_t result);
int max(int *a, int len);
int min(int *a, int len);
void cal_start_end (int M_sub, int N_sub, int width, int dia_offsets_temp, int n, int &start_temp, int &end_temp);

// free memory of taichi
void free_taichi_new(TaiChi_new *taichi_new);
void free_taichi(int num_shapes, TaiChi *taichi);
