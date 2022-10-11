#ifndef MATRIX_SPLIT_MORPHUTILS_H
#define MATRIX_SPLIT_MORPHUTILS_H

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include <sys/time.h>
#include <fstream>
#include <vector>
#include <dirent.h>
#include <chrono>
#include <cstdint>

using namespace std;
using namespace cv;

//数组转cv::Mat
Mat Array2Mat(double* array, int row, int col);
//列出目标目录下所有的文件名
int GetFileNamesInDir(char *DirPath,char *FileExtName,char FileNames[][128],int *FileNum,int MaxFileNum);
//表示一个坐标点
struct Point_t {
    int x;
    int y;
};
//移除过小或过大的连通域
void getSizeContours(vector<vector<Point>> &contours);
//细化
Mat thinImage(const cv::Mat & src, const int maxIterations = -1);
//填充
void imfill(Mat srcimage, Mat &dstimage);


//测试函数
int opencv_test();
//获得缩略图中子形状的信息
void getSdf(string path, int M, int N, int nnz, int* &csrRowIndexHostPtr, int* &csrColIndexHostPtr, float* &csrValHostPtr,
        string &category, Point_t &size,int &shapeNum, vector<Point_t> &shapePoints, vector<double> &widths);



//计时函数
/* 使用方法
 * TIMER_START(x)
 * TIMER_END(x)
 * TIMER_SEC(x)
 * */
// 用TIMER_START 定义一个变量记录开始的时间
#define TIMER_START(_X) auto _X##_start = std::chrono::steady_clock::now(), _X##_stop = _X##_start
// 用TIMER_STOP 定义一个变量记录结束的时间
#define TIMER_STOP(_X) _X##_stop = std::chrono::steady_clock::now()
// TIMER_NSEC 定义start到stop经历了多少纳秒
#define TIMER_NSEC(_X)                                                                             \
    std::chrono::duration_cast<std::chrono::nanoseconds>(_X##_stop - _X##_start).count()
// TIMER_USEC 定义start到stop历经多少微秒
#define TIMER_USEC(_X)                                                                             \
    std::chrono::duration_cast<std::chrono::microseconds>(_X##_stop - _X##_start).count()
// TIMER_MSEC 定义start到stop经历多少毫秒
#define TIMER_MSEC(_X)                                                                             \
    (0.000001 *                                                                                    \
     std::chrono::duration_cast<std::chrono::nanoseconds>(_X##_stop - _X##_start).count())
// TIMER_SEC 定义start到stop经历多少秒
#define TIMER_SEC(_X)                                                                              \
    (0.000001 *                                                                                    \
     std::chrono::duration_cast<std::chrono::microseconds>(_X##_stop - _X##_start).count())
// TIMER_MIN 定义start到stop经历多少分钟
#define TIMER_MIN(_X)                                                                              \
    std::chrono::duration_cast<std::chrono::minutes>(_X##_stop - _X##_start).count()

#endif //MATRIX_SPLIT_MORPHUTILS_H
