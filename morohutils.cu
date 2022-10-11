#include <omp.h>
#include "morphutils.h"
#include <cusp/system/cuda/arch.h>
#include <iostream>
using namespace std;
#define THREADS_PER_BLOCK 256

// list all files
int GetFileNamesInDir(char *DirPath, char *FileExtName, char FileNames[][128], int *FileNum, int MaxFileNum)
{
    DIR *dir;
    struct dirent *ptr;

    if ((dir = opendir(DirPath)) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    char *CurFileExtName = NULL;
    while ((ptr = readdir(dir)) != NULL)
    {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) ///current dir OR parrent dir
            continue;                                                        
        else if (ptr->d_type == 8)                                           ///d_type=8 is file
        {
            // char *rindex(const char *s, int c); 
            // rindex() is used to find the address of last c in string s
            CurFileExtName = rindex(ptr->d_name, '.'); 
            if (CurFileExtName != NULL && strcmp(CurFileExtName, FileExtName) == 0)
            {
                if (*FileNum < MaxFileNum)
                {
                    size_t s = sizeof(ptr->d_name);
                    memcpy(FileNames[(*FileNum)++], ptr->d_name, s);
                    //printf("CurFilePath=%s/%s\n",DirPath,ptr->d_name);
                }
            }
        }
    }
    closedir(dir);

    return 1;
}

/**
 * mathematics morphology
 */

// convert array to cv::Mat
Mat Array2Mat(double *array, int row, int col)
{
    Mat img2 = Mat(row, col, CV_8UC1);

    uchar *ptmp = NULL;

    for (int i = 0; i < row; i++)
    {
        ptmp = img2.ptr<uchar>(i); 
        for (int j = 0; j < col; j++)
        {
            if (array[i * col + j] != 0)
                ptmp[j] = (char)1;
            else
                ptmp[j] = (char)0;
        }
    }

    return img2;
}

// filling
void imfill(Mat srcimage, Mat &dstimage)
{
    Size m_Size = srcimage.size();
    Mat temimage = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcimage.type()); // expanding image
    srcimage.copyTo(temimage(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));
    floodFill(temimage, Point(0, 0), Scalar(255));
    Mat cutImg;
    // crop the expanded image
    temimage(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);
    dstimage = srcimage | (~cutImg);
}

static void thinningIteration(Mat img, int iter)
{
    Mat marker = Mat::zeros(img.size(), CV_8UC1);
    for (int i = 1; i < img.rows - 1; i++)
    {
        for (int j = 1; j < img.cols - 1; j++)
        {
            uchar p2 = img.at<uchar>(i - 1, j);
            uchar p3 = img.at<uchar>(i - 1, j + 1);
            uchar p4 = img.at<uchar>(i, j + 1);
            uchar p5 = img.at<uchar>(i + 1, j + 1);
            uchar p6 = img.at<uchar>(i + 1, j);
            uchar p7 = img.at<uchar>(i + 1, j - 1);
            uchar p8 = img.at<uchar>(i, j - 1);
            uchar p9 = img.at<uchar>(i - 1, j - 1);

            int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                    (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                    (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                    (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i, j) = 1;
        }
    }
    img &= ~marker;
}

// Apply the thinning procedure to a given image
void thinning(InputArray input, OutputArray output)
{
    Mat processed = input.getMat().clone();
    // Enforce the range of the input image to be in between 0 - 255
    processed /= 255;
    Mat prev = Mat::zeros(processed.size(), CV_8UC1);
    Mat diff;
    do
    {
        thinningIteration(processed, 0);
        thinningIteration(processed, 1);
        absdiff(processed, prev, diff);
        processed.copyTo(prev);
    } while (countNonZero(diff) > 0);

    processed *= 255;

    //output.assign(processed);
}

// remove too small or too large contour
void getSizeContours(vector<vector<Point>> &contours)
{
    int cmin = 80; // minimal lenght
    vector<vector<Point>>::const_iterator itc = contours.begin();
    while (itc != contours.end())
    {
        if ((itc->size()) < cmin)
        {
            itc = contours.erase(itc);
        }
        else
            ++itc;
    }
}

__global__ void getSdf_GPU1(
    int M,
    int pixel_on_M_block,
    int pixel_on_N_block,
    int thumbnailHeight,
    int thumbnailWidth,
    int step,
    int rows_per_thread,
    int *csrRowIndexHostPtr,
    int *csrColIndexHostPtr,
    uchar *data)
{

    int thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    int rows = thread_id * rows_per_thread;
    for (int i = rows; i < rows + rows_per_thread && i < M; i++)
    {
        for (int k = csrRowIndexHostPtr[i]; k < csrRowIndexHostPtr[i + 1]; ++k)
        {
            int x = i / pixel_on_M_block;
            int y = csrColIndexHostPtr[k] / pixel_on_N_block;
            if (x < thumbnailHeight & y < thumbnailWidth)
            {
                ((uchar *)(data + step * x))[y] = 0;
            }
        }
    }
}

void getSdf(string path, int M, int N, int nnz, int *&csrRowIndexHostPtr, int *&csrColIndexHostPtr, float *&csrValHostPtr,
            string &category, Point_t &size, int &shapeNum, vector<Point_t> &shapePoints,
            vector<double> &widths)
{
    category = "diagonal";

    string filename = path.substr(path.find_last_of('/') + 1);

    // start generating thumbnail
    int thumbnailHeight = 512;
    if (M <= thumbnailHeight)
    {
        printf("the matrix is too small! exit\n");
        return;
    }
    int pixel_on_M_block = M / thumbnailHeight;
    double scale = (double)thumbnailHeight / (double)M;
    int thumbnailWidth = (int)(scale * N);
    int pixel_on_N_block = N / thumbnailWidth;
    Mat mat(thumbnailHeight, thumbnailWidth, CV_8UC1, Scalar(255));
    // Mat mat1(thumbnailHeight, thumbnailWidth, CV_8UC1, Scalar(255));

    int *d_csrRowIndexHostPtr;
    int *d_csrColIndexHostPtr;

    cudaMalloc(((void **)(&d_csrRowIndexHostPtr)), (M + 1) * sizeof(int));
    cudaMalloc(((void **)(&d_csrColIndexHostPtr)), nnz * sizeof(int));

    cudaMemcpy(d_csrRowIndexHostPtr, csrRowIndexHostPtr, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIndexHostPtr, csrColIndexHostPtr, nnz * sizeof(int), cudaMemcpyHostToDevice);

    int size_m = thumbnailHeight; // (M - 1) / pixel_on_M_block;
    int size_n = thumbnailWidth;  //(N - 1) / pixel_on_N_block;
    uchar *d_data;
    cudaMalloc(((void **)(&d_data)), size_m * size_n * sizeof(uchar));
    cudaMemcpy(d_data, mat.data, size_m * size_n * sizeof(uchar), cudaMemcpyHostToDevice);

    int NUM_BLOCKS = cusp::system::cuda::detail::max_active_blocks(getSdf_GPU1, THREADS_PER_BLOCK, 0);
    NUM_BLOCKS = min(NUM_BLOCKS, (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    NUM_BLOCKS = max(1, NUM_BLOCKS);

    int rows_per_thread = (M + NUM_BLOCKS * THREADS_PER_BLOCK - 1) / (NUM_BLOCKS * THREADS_PER_BLOCK);

    getSdf_GPU1<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(M, pixel_on_M_block, pixel_on_N_block, thumbnailHeight, thumbnailWidth, mat.step.p[0], rows_per_thread, d_csrRowIndexHostPtr, d_csrColIndexHostPtr, d_data);
    cudaMemcpy(mat.data, d_data, size_m * size_n * sizeof(uchar), cudaMemcpyDeviceToHost);
    cudaFree(d_csrRowIndexHostPtr);
    cudaFree(d_csrColIndexHostPtr);
    cudaFree(d_data);

    Mat originMat = mat;
    Mat rev_mat = 255 - mat;
    Mat rev_mat_filled;
    
    imfill(rev_mat, rev_mat_filled);

    // two times opening operation, get rectangle
    Mat rev_mat_filled_open, rev_mat_filled_open_open;
    Mat elementHorizonLine = getStructuringElement(MORPH_RECT, Size(30, 1));
    morphologyEx(rev_mat_filled, rev_mat_filled_open, MORPH_OPEN, elementHorizonLine);

    Mat elementVerticalLine = getStructuringElement(MORPH_RECT, Size(1, 30));
    morphologyEx(rev_mat_filled_open, rev_mat_filled_open_open, MORPH_OPEN, elementVerticalLine);

    Mat rev_delete_square = rev_mat - rev_mat_filled_open_open;

    Mat &rev_delete_sqaure_thin = rev_delete_square;

    Mat rev_delete_sqaure_thin_open;
    Mat elementDiag = Mat::eye(5, 5, CV_8U);
    morphologyEx(rev_delete_sqaure_thin, rev_delete_sqaure_thin_open, MORPH_OPEN, elementDiag);

    Mat &rev_delete_sqaure_thin_open_dilate = rev_delete_sqaure_thin_open;

    // get connected areas
    vector<vector<Point>> contours;
    findContours(rev_delete_sqaure_thin_open_dilate, contours,
                 RETR_CCOMP, CHAIN_APPROX_NONE, Point(0, 0));

    // filter too small connected areas
    getSizeContours(contours);

    size.x = thumbnailHeight;
    size.y = thumbnailWidth;
    shapeNum = contours.size();

    for (auto &contour : contours)
    {
        Point left_up_corner = contour[0];
        Point right_bottom_corner = contour[contour.size() / 2];

        double area = cv::contourArea(contour);
        double distance = (left_up_corner.x - right_bottom_corner.x) * (left_up_corner.x - right_bottom_corner.x) + (left_up_corner.y - right_bottom_corner.y) * (left_up_corner.y - right_bottom_corner.y);
        distance = sqrt(distance);
        widths.push_back(area / distance);

        Point_t left_up_corner_point{left_up_corner.x, left_up_corner.y};
        Point_t right_bottom_corner_point{right_bottom_corner.x, right_bottom_corner.y};

        shapePoints.push_back(left_up_corner_point);
        shapePoints.push_back(right_bottom_corner_point);
    }

    // output sdf 
    // cout << thumbnailHeight << " " << thumbnailWidth << " " << shapeNum << endl;
    // for(int i=0;i<shapeNum;++i) 
    // {
    //     cout << "diagnonal" << " " << shapePoints[2*i].x << " " << shapePoints[2*i].y << " " 
    //          << shapePoints[2*i+1].x << " " << shapePoints[2*i+1].y << "\n";
    // }
}
