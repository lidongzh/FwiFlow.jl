// Dongzhuo Li 04/22/2018

#include "Parameter.h"
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace rapidjson;

Parameter::Parameter() {
  cout << "ERROR: You need to input parameter file name!" << endl;
  exit(1);
}

Parameter::Parameter(const std::string &para_fname) {
  string line;
  ifstream parafile;

  parafile.open(para_fname);

  if (!parafile.is_open()) {
    cout << "Error opening parameter file" << endl;
    exit(1);
  }

  // read the whole line of json file
  getline(parafile, line);
  // cout << line << endl;
  parafile.close();

  Document json_para;
  json_para.Parse<0>(line.c_str());

  assert(json_para.IsObject());
  // assert(para.HasMember("nx"));
  // assert(para["nx"].IsInt());

  cout << "----------Reading Parameter File------------" << endl;

  assert(json_para.HasMember("nz"));
  assert(json_para["nz"].IsInt());
  nz_ = json_para["nz"].GetInt();
  cout << "	nz = " << nz_ << endl;

  assert(json_para.HasMember("nx"));
  assert(json_para["nx"].IsInt());
  nx_ = json_para["nx"].GetInt();
  cout << "	nx = " << nx_ << endl;

  // assert(json_para.HasMember("dh"));
  // assert(json_para["dh"].IsDouble());
  // dh_ = json_para["dh"].GetDouble();
  // cout << "	dh = " << dh_ << endl;

  assert(json_para.HasMember("dz"));
  assert(json_para["dz"].IsNumber());
  dz_ = json_para["dz"].GetDouble();
  cout << "	dz = " << dz_ << endl;

  assert(json_para.HasMember("dx"));
  assert(json_para["dx"].IsNumber());
  dx_ = json_para["dx"].GetDouble();
  cout << "	dx = " << dx_ << endl;

  assert(json_para.HasMember("nSteps"));
  assert(json_para["nSteps"].IsInt());
  nSteps_ = json_para["nSteps"].GetInt();
  cout << "	nSteps = " << nSteps_ << endl;

  assert(json_para.HasMember("nPoints_pml"));
  assert(json_para["nPoints_pml"].IsInt());
  nPoints_pml_ = json_para["nPoints_pml"].GetInt();
  cout << "	nPml = " << nPoints_pml_ << endl;

  assert(json_para.HasMember("nPad"));
  assert(json_para["nPad"].IsInt());
  nPad_ = json_para["nPad"].GetInt();
  cout << "	nPad = " << nPad_ << endl;

  assert(json_para.HasMember("dt"));
  assert(json_para["dt"].IsDouble());
  dt_ = json_para["dt"].GetDouble();
  cout << "	dt = " << dt_ << endl;

  assert(json_para.HasMember("f0"));
  // assert(json_para["f0"].IsNumber());
  f0_ = json_para["f0"].GetDouble();
  cout << "	f0 = " << f0_ << endl;

  assert(json_para.HasMember("Cp_fname"));
  assert(json_para["Cp_fname"].IsString());
  Cp_fname_ = json_para["Cp_fname"].GetString();
  cout << "	Cp_fname = " << Cp_fname_ << endl;

  assert(json_para.HasMember("Cp_fname"));
  assert(json_para["Cs_fname"].IsString());
  Cs_fname_ = json_para["Cs_fname"].GetString();
  cout << "	Cs_fname = " << Cs_fname_ << endl;

  assert(json_para.HasMember("Den_fname"));
  assert(json_para["Den_fname"].IsString());
  Den_fname_ = json_para["Den_fname"].GetString();
  cout << "	Den_fname = " << Den_fname_ << endl;

  // assert(json_para.HasMember("src_rec_pos_fname"));
  // assert(json_para["src_rec_pos_fname"].IsString());
  // src_rec_pos_fname_ = json_para["src_rec_pos_fname"].GetString();
  // cout << "	src_rec_pos_fname = " << src_rec_pos_fname_ << endl;

  assert(json_para.HasMember("data_dir_name"));
  assert(json_para["data_dir_name"].IsString());
  data_dir_name_ = json_para["data_dir_name"].GetString();
  cout << "	data_dir_name = " << data_dir_name_ << endl;

  assert(json_para.HasMember("isAc"));
  assert(json_para["isAc"].IsBool());
  isAc_ = json_para["isAc"].GetBool();
  cout << "	Acoustic = " << isAc_ << endl;

  assert(json_para.HasMember("withAdj"));
  assert(json_para["withAdj"].IsBool());
  withAdj_ = json_para["withAdj"].GetBool();
  cout << "	With Adjoint Computation = " << withAdj_ << endl;

  assert(json_para.HasMember("if_res"));
  assert(json_para["if_res"].IsBool());
  if_res_ = json_para["if_res"].GetBool();
  cout << "	With Residual Computation = " << if_res_ << endl;

  assert(json_para.HasMember("if_win"));
  assert(json_para["if_win"].IsBool());
  if_win_ = json_para["if_win"].GetBool();
  cout << "	With Window Selection = " << if_win_ << endl;

  if_filter_ = false;
  if (json_para.HasMember("if_filter")) {
    assert(json_para["if_filter"].IsBool());
    if_filter_ = json_para["if_filter"].GetBool();
  }
  cout << "	With Filtering = " << if_filter_ << endl;

  if (if_filter_ == true) {
    assert(json_para.HasMember("filter"));
    assert(json_para["filter"].IsArray());
    const Value &js_filter = json_para["filter"];
    for (SizeType ii = 0; ii < js_filter.Size(); ii++) {
      filter_[ii] = js_filter[ii].GetDouble();
    }
    printf("	filter = [%.2f, %.2f, %.2f, %.2f]\n", filter_[0], filter_[1],
           filter_[2], filter_[3]);
  }

  assert(json_para.HasMember("if_src_update"));
  assert(json_para["if_src_update"].IsBool());
  if_src_update_ = json_para["if_src_update"].GetBool();
  cout << "	With Source Update = " << if_src_update_ << endl;

  // turn on residual computation if compute adjoint
  if (withAdj_ == true) {
    if_res_ = true;
  }
}

Parameter::Parameter(const std::string &para_fname, int calc_id) {
  string line;
  ifstream parafile;

  parafile.open(para_fname);

  if (!parafile.is_open()) {
    cout << "Error opening parameter file" << endl;
    exit(1);
  }

  // read the whole line of json file
  getline(parafile, line);
  // cout << line << endl;
  parafile.close();

  Document json_para;
  json_para.Parse<0>(line.c_str());

  assert(json_para.IsObject());

  cout << "----------Reading Parameter File------------" << endl;

  assert(json_para.HasMember("nz"));
  assert(json_para["nz"].IsInt());
  nz_ = json_para["nz"].GetInt();
  cout << "	nz = " << nz_ << endl;

  assert(json_para.HasMember("nx"));
  assert(json_para["nx"].IsInt());
  nx_ = json_para["nx"].GetInt();
  cout << "	nx = " << nx_ << endl;

  assert(json_para.HasMember("dz"));
  assert(json_para["dz"].IsNumber());
  dz_ = json_para["dz"].GetDouble();
  cout << "	dz = " << dz_ << endl;

  assert(json_para.HasMember("dx"));
  assert(json_para["dx"].IsNumber());
  dx_ = json_para["dx"].GetDouble();
  cout << "	dx = " << dx_ << endl;

  assert(json_para.HasMember("nSteps"));
  assert(json_para["nSteps"].IsInt());
  nSteps_ = json_para["nSteps"].GetInt();
  cout << "	nSteps = " << nSteps_ << endl;

  assert(json_para.HasMember("nPoints_pml"));
  assert(json_para["nPoints_pml"].IsInt());
  nPoints_pml_ = json_para["nPoints_pml"].GetInt();
  cout << "	nPml = " << nPoints_pml_ << endl;

  assert(json_para.HasMember("nPad"));
  assert(json_para["nPad"].IsInt());
  nPad_ = json_para["nPad"].GetInt();
  cout << "	nPad = " << nPad_ << endl;

  assert(json_para.HasMember("dt"));
  assert(json_para["dt"].IsDouble());
  dt_ = json_para["dt"].GetDouble();
  cout << "	dt = " << dt_ << endl;

  assert(json_para.HasMember("f0"));
  // assert(json_para["f0"].IsNumber());
  f0_ = json_para["f0"].GetDouble();
  cout << "	f0 = " << f0_ << endl;

  assert(json_para.HasMember("survey_fname"));
  assert(json_para["survey_fname"].IsString());
  survey_fname_ = json_para["survey_fname"].GetString();
  cout << "	survey_fname = " << survey_fname_ << endl;

  assert(json_para.HasMember("data_dir_name"));
  assert(json_para["data_dir_name"].IsString());
  data_dir_name_ = json_para["data_dir_name"].GetString();
  cout << "	data_dir_name = " << data_dir_name_ << endl;

  if (json_para.HasMember("scratch_dir_name")) {
    if_save_scratch_ = true;
    assert(json_para["scratch_dir_name"].IsString());
    scratch_dir_name_ = json_para["scratch_dir_name"].GetString();
    cout << "	scratch_dir_name = " << scratch_dir_name_ << endl;
  }

  assert(json_para.HasMember("isAc"));
  assert(json_para["isAc"].IsBool());
  isAc_ = json_para["isAc"].GetBool();
  cout << "	Acoustic = " << isAc_ << endl;

  if (calc_id == 0) {
    if_res_ = true;
    withAdj_ = false;
  } else if (calc_id == 1) {
    if_res_ = true;
    withAdj_ = true;
  } else if (calc_id == 2) {
    if_res_ = false;
    withAdj_ = false;
  } else {
    printf("invalid calc_id mode!\n");
    exit(1);
  }

  assert(json_para.HasMember("if_win"));
  assert(json_para["if_win"].IsBool());
  if_win_ = json_para["if_win"].GetBool();
  cout << "	With Window Selection = " << if_win_ << endl;

  if_filter_ = false;
  if (json_para.HasMember("if_filter")) {
    assert(json_para["if_filter"].IsBool());
    if_filter_ = json_para["if_filter"].GetBool();
  }
  cout << "	With Filtering = " << if_filter_ << endl;

  if (if_filter_ == true) {
    assert(json_para.HasMember("filter"));
    assert(json_para["filter"].IsArray());
    const Value &js_filter = json_para["filter"];
    for (SizeType ii = 0; ii < js_filter.Size(); ii++) {
      filter_[ii] = js_filter[ii].GetDouble();
    }
    printf("	filter = [%.2f, %.2f, %.2f, %.2f]\n", filter_[0], filter_[1],
           filter_[2], filter_[3]);
  }

  assert(json_para.HasMember("if_src_update"));
  assert(json_para["if_src_update"].IsBool());
  if_src_update_ = json_para["if_src_update"].GetBool();
  cout << "	With Source Update = " << if_src_update_ << endl;

  // turn on residual computation if compute adjoint
  if (withAdj_ == true) {
    if_res_ = true;
  }
}

Parameter::~Parameter() {}