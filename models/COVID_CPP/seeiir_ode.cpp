//#include <boost/math/distributions/negative_binomial.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>      /* printf */
#include <math.h>       /* lgamma */
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
namespace py = pybind11;

double hazard_func(double dt, int A, double d_beta, double mat[][8], 
double I1[], double I2[], double r0s, double r_init, double gamma, int a){
  
  
  double prod = 1;
  double b_mat[8][8];
  for (int i=0; i < A; ++i) 
    for (int j=0; j < A; ++j)
    {
        {
          b_mat[i][j] = exp(d_beta)* (r_init/r0s)* mat[i][j];
        }
    }
  
  for (int i=0; i < A; ++i)
      prod *= pow( (1-(b_mat[a][i])), (I1[i] + I2[i]) );
      
  return (1 - prod)*dt;

}

void SEEIIR_ODE(py::array_t<double> beta, py::array_t<double> lamda, py::array_t<double> i1,
py::array_t<double> i2, py::array_t<double> e1, py::array_t<double> e2,
py::array_t<double> r, py::array_t<double> s, py::array_t<double> mix_mat, 
py::array_t<double> m_prelock, py::array_t<double> m_postlock, double r0_init, double r0star, 
int Tlock, double gamma, double sigma, double dt, int T, int A){
  
  auto beta_ = beta.mutable_unchecked<1>(); 
  auto lamda_ = lamda.mutable_unchecked<2>();
  auto i1_ = i1.mutable_unchecked<2>(); 
  auto i2_ = i2.mutable_unchecked<2>(); 
  auto e1_ = e1.mutable_unchecked<2>(); 
  auto e2_ = e2.mutable_unchecked<2>(); 
  auto r_ = r.mutable_unchecked<2>(); 
  auto s_ = s.mutable_unchecked<2>(); 
  auto mix_mat_ = mix_mat.mutable_unchecked<3>(); 
  auto m_prelock_ = m_prelock.mutable_unchecked<2>(); 
  auto m_postlock_ = m_postlock.mutable_unchecked<2>(); 
  
  
  double m_mat[8][8];
  double I1_mat[8];
  double I2_mat[8];
  int numRows = A;
    for (ssize_t i = 1; i < s_.shape(0); i++)
        for (ssize_t a = 0; a < s_.shape(1); a++)
        {
          if(i < Tlock + 1)
           {
              for (ssize_t row=0; row < m_prelock_.shape(0); ++row) 
                for (ssize_t col=0; col < m_prelock_.shape(1); ++col)
                {
                    {
                      m_mat[row][col] = mix_mat_(row,col,i-1)*m_prelock_(row,col);
                    }
                }              
              for (ssize_t col=0; col < i1_.shape(1); ++col) 
                {
                  I1_mat[col] = i1_(i-1,col);
                  I2_mat[col] = i2_(i-1,col);
                }
              double d_beta = beta_(i-1);
              lamda_(i-1,a) = hazard_func(dt, A, d_beta, m_mat, I1_mat, I2_mat, r0star, r0_init, gamma, a);
              s_(i,a) = s_(i-1,a)*(1 - lamda_(i-1,a));
              e1_(i,a) = (e1_(i-1,a)*(1- sigma*dt)) + (s_(i-1,a)*lamda_(i-1,a));
           }
           else
           {
              for (ssize_t row=0; row < m_postlock_.shape(0); ++row) 
                for (ssize_t col=0; col < m_postlock_.shape(1); ++col)
                {
                    {
                      m_mat[row][col] = mix_mat_(row,col,i-1)*m_postlock_(row,col);
                    }
                }              
              for (ssize_t col=0; col < i1_.shape(1); ++col) 
                {
                  I1_mat[col] = i1_(i-1,col);
                  I2_mat[col] = i2_(i-1,col);
                }
              double d_beta = beta_(i-1);
              lamda_(i-1,a) = hazard_func(dt, A, d_beta, m_mat, I1_mat, I2_mat, r0star, r0_init, gamma, a);
              s_(i,a) = s_(i-1,a)*(1 - lamda_(i-1,a));
              e1_(i,a) = (e1_(i-1,a)*(1- sigma*dt)) + (s_(i-1,a)*lamda_(i-1,a));
           }
          e2_(i,a) = (e2_(i-1,a)*(1- sigma*dt)) + (e1_(i-1,a)*sigma*dt);
          i1_(i,a) = (i1_(i-1,a)*(1- gamma*dt)) + (e2_(i-1,a)*sigma*dt);
          i2_(i,a) = (i2_(i-1,a)*(1- gamma*dt)) + (i1_(i-1,a)*gamma*dt);
          r_(i,a) =  r_(i-1,a) + (i2_(i-1,a)*gamma*dt);
        }          
}

void convolution_2D(py::array_t<double> delta, py::array_t<double> conv_series,
py::array_t<double> cdf_dead, int conlen){

  auto cdf_dead_ = cdf_dead.mutable_unchecked<1>(); 
  auto delta_ = delta.mutable_unchecked<2>();
  auto conv_series_ = conv_series.mutable_unchecked<2>(); 

  for (int n = 0; n < delta_.shape(0); n++)
    for (int a = 0; a < delta_.shape(1); a++)
      {
        double conv_temp = 0.0;
        int range = std::min(conlen,n);
        for(int k=0; k<=range; k++)
            conv_temp += delta_(n-k,a)*cdf_dead_(k);
        conv_series_(n,a) = conv_temp;
      }
}

PYBIND11_PLUGIN(seeiir_ode) {
    pybind11::module m("seeiir_ode", "auto-compiled c++ extension");
    m.def("SEEIIR_ODE", &SEEIIR_ODE, py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
    py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
    py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
    py::arg().noconvert(), py::arg().noconvert(), py::arg(), py::arg(), py::arg(), py::arg(),
    py::arg(), py::arg(), py::arg(), py::arg());
    m.def("convolution_2D", &convolution_2D, py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
    py::arg());
    return m.ptr();
}
