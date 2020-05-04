#include <vector>
#include <algorithm>
#include <Eigen/Core>

#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <vector>
#include <iostream>
#include <utility>  
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;


double solution(double a, double b, double f, double h){
  double d = fabs(a-b);
  if(d>=f*h) 
    return std::min(a, b) + f*h;
  else 
    return (a+b+sqrt(2*f*f*h*h-(a-b)*(a-b)))/2;
}

void sweep(double *u, 
    const std::vector<int>& I, const std::vector<int>& J,
    const double *f, int m, int n, double h, int ix, int jx){
    for(int i: I){
      for(int j:J){
        if (i==ix && j==jx) continue; 
        double a, b;
        if (i==0){
          a = u[j*(m+1)+1];
        }
        else if (i==m){
          a = u[j*(m+1)+m-1];
        }
        else{
          a = std::min(u[j*(m+1)+i+1], u[j*(m+1)+i-1]);
        }
        if (j==0){
          b = u[(m+1)+i];
        }
        else if (j==n){
          b = u[(n-1)*(m+1)+i];
        }
        else{
          b = std::min(u[(j-1)*(m+1)+i], u[(j+1)*(m+1)+i]);
        }
        u[j*(m+1)+i] = solution(a, b, f[j*(m+1)+i], h);
      }
    }
    
}

void forward(double *u, const double *f, int m, int n, double h, int ix, int jx){
  for(int i=0;i<m+1;i++){
    for(int j=0;j<n+1;j++){
      u[j*(m+1)+i] = 100000.0;
      if (i==ix && j==jx) u[j*(m+1)+i] = 0.0;
    }
  }
  std::vector<int> I, J, iI, iJ;
  for(int i=0;i<m+1;i++) {
    I.push_back(i);
    iI.push_back(m-i);
  }
  for(int i=0;i<n+1;i++) {
    J.push_back(i);
    iJ.push_back(n-i);
  }

  Eigen::VectorXd uvec_old = Eigen::Map<const Eigen::VectorXd>(u, (m+1)*(n+1)), uvec;
  bool converged = false;
  for(int i = 0;i<100;i++){
    sweep(u, I, J, f, m, n, h, ix, jx);
    sweep(u, iI, J, f, m, n, h, ix, jx);
    sweep(u, iI, iJ, f, m, n, h, ix, jx);
    sweep(u, I, iJ, f, m, n, h, ix, jx);
    uvec = Eigen::Map<const Eigen::VectorXd>(u, (m+1)*(n+1));
    double err = (uvec-uvec_old).norm()/uvec_old.norm();
    // printf("ERROR AT ITER %d: %g\n", i, err);
    if (err < 1e-8){ 
      converged = true;
      break; 
    }
    uvec_old = uvec;
  }

  if(!converged){
    printf("ERROR: Eikonal does not converge!\n");
  }
  

}

void backward(
  double *grad_f, 
  const double * grad_u,
  const double *u, const double *f, int m, int n, double h, int ix, int jx){

    Eigen::VectorXd dFdf((m+1)*(n+1));
    for (int i=0;i<(m+1)*(n+1);i++){
      dFdf[i] = -2*f[i]*h*h;
    }
    dFdf[jx*(m+1)+ix] = 0.0;
    std::vector<T> triplets;

    for(int j=0;j<n+1;j++){
      for(int i=0;i<m+1;i++){
        int idx = j*(m+1)+i;
        if (i==ix && j==jx) {
          triplets.push_back(T(idx, idx, 1.0));
          continue;
        }

        // double val = 0.0;
        if(i==0){
          if(u[idx]>u[j*(m+1)+1]){
            triplets.push_back(T(idx, idx, 2*(u[idx]-u[j*(m+1)+1]) ));
            triplets.push_back(T(idx, j*(m+1)+1, 2*(u[j*(m+1)+1]-u[idx]) ));

            // val += (u[idx]-u[j*(m+1)+1])*(u[idx]-u[j*(m+1)+1]);
          }
        }
        else if (i==m){

          if(u[idx]>u[j*(m+1)+m-1]){
            triplets.push_back(T(idx, idx, 2*(u[idx]-u[j*(m+1)+m-1]) ));
            triplets.push_back(T(idx, j*(m+1)+m-1, 2*(u[j*(m+1)+m-1]-u[idx]) ));

            // val += (u[idx]-u[j*(m+1)+m-1])*(u[idx]-u[j*(m+1)+m-1]);
          }

        }
        else {

          double a = u[j*(m+1)+i+1]>u[j*(m+1)+i-1] ? u[j*(m+1)+i-1] : u[j*(m+1)+i+1];
          if (u[idx]>a){
            triplets.push_back(T(idx, idx, 2*(u[idx]-a) ));
            if (u[j*(m+1)+i+1]>u[j*(m+1)+i-1])
              triplets.push_back(T(idx, j*(m+1)+i-1, 2*(a-u[idx]) ));
            else
              triplets.push_back(T(idx, j*(m+1)+i+1, 2*(a-u[idx]) ));

            // val += (a-u[idx])*(a-u[idx]);
          }

        }

        if (j==0){
          if (u[idx]>u[m+1+i]){
            triplets.push_back(T(idx, idx, 2*(u[idx]-u[m+1+i]) ));
            triplets.push_back(T(idx, (m+1)+i, 2*(u[m+1+i]-u[idx]) ));

            // val += (u[idx]-u[m+1+i])*(u[idx]-u[m+1+i]);
          }

        }
        else if(j==n){

          if (u[idx]>u[(n-1)*(m+1)+i]){
            triplets.push_back(T(idx, idx, 2*(u[idx]-u[(n-1)*(m+1)+i]) ));
            triplets.push_back(T(idx, (n-1)*(m+1)+i, 2*(u[(n-1)*(m+1)+i]-u[idx]) ));

            // val += (u[idx]-u[(n-1)*(m+1)+i])*(u[idx]-u[(n-1)*(m+1)+i]);
          }

        }
        else {
          double b = u[(j+1)*(m+1)+i]>u[(j-1)*(m+1)+i] ? u[(j-1)*(m+1)+i] : u[(j+1)*(m+1)+i];
          if (u[idx]>b){
            triplets.push_back(T(idx, idx, 2*(u[idx]-b) ));
            if (u[(j+1)*(m+1)+i]>u[(j-1)*(m+1)+i])
              triplets.push_back(T(idx, (j-1)*(m+1)+i, 2*(b-u[idx]) ));
            else
              triplets.push_back(T(idx, (j+1)*(m+1)+i, 2*(b-u[idx]) ));

              // val += (b-u[idx])*(b-u[idx]);
          }
        }

        // val -= f[idx]*f[idx]*h*h;
        // printf("VAL = %g\n", val);
        
      }
    }

    SpMat A((m+1)*(n+1), (m+1)*(n+1));
    A.setFromTriplets(triplets.begin(), triplets.end());
    A = A.transpose();
    Eigen::SparseLU<SpMat> solver;
     
    Eigen::VectorXd g = Eigen::Map<const Eigen::VectorXd>(grad_u, (m+1)*(n+1));
    solver.analyzePattern(A);
    solver.factorize(A);
    Eigen::VectorXd res = solver.solve(g);
    for(int i=0;i<(m+1)*(n+1);i++){
      grad_f[i] = -res[i] * dFdf[i];
    }
    
}
