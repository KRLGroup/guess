#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace guess {

  /*! real type of the whole system */
  typedef float real;

  template <int Dim_>
    using Vector_ = Eigen::Matrix<real, Dim_, 1>;
  template <int Rows_, int Cols_>
    using Matrix_ = Eigen::Matrix<real, Rows_, Cols_>;
  template <int Dim_>
    using Isometry_ = Eigen::Transform<real, Dim_, Eigen::Isometry>;
  
  /*! Eigen::Vector of type real */
  typedef Vector_<Eigen::Dynamic> VectorX;
  typedef Vector_<2> Vector2;
  typedef Vector_<3> Vector3;
  typedef Vector_<4> Vector4;
  typedef Vector_<5> Vector5;
  typedef Vector_<6> Vector6;
  typedef Vector_<7> Vector7;

  /*! std::vector of Vector with aligned_allocation */
  typedef std::vector<Vector2,
    Eigen::aligned_allocator<Vector2> > Vector2Vector;
  typedef std::vector<Vector6,
    Eigen::aligned_allocator<Vector6> > Vector6Vector;
  
  /*! Eigen::Matrix of type real */
  typedef Matrix_<Eigen::Dynamic, Eigen::Dynamic> MatrixX;
  typedef Matrix_<2, 2> Matrix2;
  typedef Matrix_<3, 3> Matrix3;
  typedef Matrix_<4, 4> Matrix4;
  typedef Matrix_<5, 5> Matrix5;
  typedef Matrix_<6, 6> Matrix6;
    
  /*! Eigen::Isometry of type real */
  typedef Isometry_<2> Isometry2;
  typedef Isometry_<3> Isometry3;

  /*! Eigen::AngleAxis of type real */
  typedef Eigen::AngleAxis<real> AngleAxis;

  /*! Eigen::Quaternion of type real */
  typedef Eigen::Quaternion<real> Quaternion;

  /*! Eigen::Matrix (complex) of type real */
  typedef Eigen::Matrix< std::complex<real> , Eigen::Dynamic , 1> VectorXc;
  typedef Eigen::Matrix< std::complex<real> , Eigen::Dynamic , Eigen::Dynamic> MatrixXc;
  
}
