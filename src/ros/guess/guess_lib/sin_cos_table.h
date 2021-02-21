#pragma once
#include "def.h"

namespace guess {
  class SinCosTable{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    SinCosTable(){
      _initialized = false;
      _angle_min = 0.f;
      _angle_increment = 0.f;      
    }

    inline void init(const real& angle_min,
                     const real& angle_increment,
                     const size_t& s) {
      if (_angle_min == angle_min &&
          _angle_increment == angle_increment &&
          _table.size() == s)
        return;
      
      _angle_min = angle_min;
      _angle_increment = angle_increment;
      _table.resize(s);
      
      for (size_t i = 0; i < s; ++i){
        real alpha = _angle_min + i * _angle_increment;
        _table[i] = Eigen::Vector2f(cos(alpha), sin(alpha));
      }
      _initialized = true;
    }
    
    inline const Vector2& sincos(int i) const {
      if(!_initialized)
        throw std::runtime_error("[SinCosTable][sincos]: missing initialization");
      return _table[i];
    }
    
  protected:
    bool _initialized;
    real _angle_min;
    real _angle_increment;
    Vector2Vector _table;
  };
}
