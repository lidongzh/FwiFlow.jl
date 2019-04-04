#include <iostream>
#include <Eigen/Sparse>
using namespace Eigen;
using namespace std;

#define RHO_o 1000.0
#define GRAV 9.8

// ------------
// | |
// |  i   --> j
// |
typedef Eigen::MatrixXd densMat;
// typedef Eigen::TensorMap<Eigen::Tensor<const double, 2, 1, long>, 16, Eigen::MakePointer> densMat;

// TODO implement interpolation of g -> a, b, c, d