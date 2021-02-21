#pragma once

#include "def.h"
#include "labels.h"

namespace guess {

  struct LaserBeam {
    LaserBeam(const Label label_ = Label::Unknown,
              const real& value_ = 0.0) {
      label = label_;
      value = value_;
    }
    real value;
    Label label;
  };

  struct LaserReading : public std::vector<LaserBeam> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    double timestamp = 0.0;
    Isometry2 pose = Isometry2::Identity();
  };
  
}
