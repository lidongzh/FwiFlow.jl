#include "ops_utils.h"

using namespace std;
using namespace rapidjson;

void fileBinLoad(float *h_bin, int size, std::string fname) {
  FILE *fp = fopen(fname.c_str(), "rb");
  if (fp == NULL) {
    printf("File reading error!\n");
    exit(1);
  } else {
    size_t sizeRead = fread(h_bin, sizeof(float), size, fp);
  }
  fclose(fp);
}

void fileBinLoadDouble(double *h_bin, int size, std::string fname) {
  FILE *fp = fopen(fname.c_str(), "rb");
  if (fp == NULL) {
    printf("File reading error!\n");
    exit(1);
  } else {
    size_t sizeRead = fread(h_bin, sizeof(double), size, fp);
  }
  fclose(fp);
}

void fileBinWrite(float *h_bin, int size, std::string fname) {
  FILE *fp = fopen(fname.c_str(), "wb");
  if (fp == NULL) {
    printf("File writing error!\n");
    exit(1);
  } else {
    fwrite(h_bin, sizeof(float), size, fp);
  }
  fclose(fp);
}

void intialArray(float *ip, int size, float value) {
  for (int i = 0; i < size; i++) {
    ip[i] = value;
    // printf("value = %f\n", value);
  }
}

void readParJson(std::string para_fname, int &nz, int &nx, int &nPml, int &nPad) {
  std::string line;
  ifstream parafile;
  parafile.open(para_fname);
  if (!parafile.is_open()) {
    std::cout << "Error opening file" << std::endl;
    exit(1);
  }
  // read the whole line of json file
  getline(parafile, line);
  // std::cout << line << std::endl;
  parafile.close();

  Document json_para;
  json_para.Parse<0>(line.c_str());

  assert(json_para.IsObject());

  assert(json_para.HasMember("nz"));
  assert(json_para["nz"].IsInt());
  nz = json_para["nz"].GetInt();
  // std::cout << " nz = " << nz << std::endl;

  assert(json_para.HasMember("nx"));
  assert(json_para["nx"].IsInt());
  nx = json_para["nx"].GetInt();
  // std::cout << " nx = " << nx << std::endl;

  assert(json_para.HasMember("nPoints_pml"));
  assert(json_para["nPoints_pml"].IsInt());
  nPml = json_para["nPoints_pml"].GetInt();
  // cout << " nPml = " << nPml << endl;

  assert(json_para.HasMember("nPad"));
  assert(json_para["nPad"].IsInt());
  nPad = json_para["nPad"].GetInt();
  // std::cout << " nPad = " << nPad << std::endl;
}

