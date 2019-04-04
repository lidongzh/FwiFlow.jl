#ifndef OPS_UTILS_H__
#define OPS_UTILS_H__

#include <stdio.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "document.h"
#include "rapidjson.h"

#define PI (3.141592653589793238462643383279502884197169)
// #define DEBUG

void fileBinLoad(float *h_bin, int size, std::string fname);

void fileBinWrite(float *h_bin, int size, std::string fname);

void fileBinLoadDouble(double *h_bin, int size, std::string fname);

void readParJson(std::string para_fname, int &nz, int &nx, int &nPml, int &nPad);

void intialArray(float *ip, int size, float value);

#endif