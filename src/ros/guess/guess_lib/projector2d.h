#pragma once

#include "laser_reading.hpp"
#include "point.hpp"
#include "sin_cos_table.h"

namespace guess {

  class Projector2d {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    struct Config {
      real min_range = 30;
      real max_range = 0.1;
      size_t num_ranges = 1024;
      real fov = 1.5 * M_PI;
    };

    Projector2d() {
      _initialized = false;
      _angle_increment = 0.f;
    }

    ~Projector2d() {}

    inline const Config& config() const {return _config;}
    inline Config& mutableConfig() {return _config;}

    inline const SinCosTable& sinCosTable() const {return _table;}
    inline SinCosTable& sinCosTable() {return _table;}

    //! @brief initializes the sinCos table
    void init();
    
    //! @brief projects the cloud into a laser reading given a pose
    void project(LaserReading& laser_reading_,
                 const Isometry2& T_,
                 const LabelCloud& cloud_);

    //! @brief unprojects the current laser reading into a cloud
    void unproject(LabelCloud& cloud_,
                   const LaserReading& laser_reading_);

  private:
    bool _initialized;
    Config _config;
    SinCosTable _table;
    real _angle_increment;
    
  };

}
