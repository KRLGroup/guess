#include "projector2d.h"

using namespace guess;

void Projector2d::init() {
  _angle_increment = _config.fov / _config.num_ranges;  
  _table.init(-_config.fov * 0.5, _angle_increment, _config.num_ranges);
  _initialized = true;
}

void Projector2d::project(LaserReading& laser_reading_,
                          const Isometry2& T_,
                          const LabelCloud& cloud_) { 
  if(!_initialized)
    throw std::runtime_error("[Projector2d][project]: missing initialization!");

  const real middle = _config.num_ranges * 0.5;
  const real inverse_angle_increment = 1. / _angle_increment;
  const Matrix2& R = T_.linear();
  laser_reading_.resize(_config.num_ranges);
  std::fill(laser_reading_.begin(), laser_reading_.end(), LaserBeam(Label::Unknown, 1e3));
  for (size_t i = 0; i < cloud_.size(); ++i) {
    const LabPoint& rp = cloud_[i];
    Vector2 p = T_ * rp.p.head<2>();
    real angle = atan2(p.y(), p.x());
    int bin = round(angle * inverse_angle_increment + middle);

    if (bin < 0 || bin >= _config.num_ranges)
      continue;
    real range = p.norm();
    Label label = rp.label;
    real& brange = laser_reading_[bin].value;
    Label& blabel = laser_reading_[bin].label;
    
    //bdc z-buffer
    if (brange > range) {
      brange = range;
      blabel = label;
    }
    else 
      continue;      
  }

}

void Projector2d::unproject(LabelCloud& cloud_,
                            const LaserReading& laser_reading_) {
  if(!_initialized)
    throw std::runtime_error("[Projector2d][unproject]: missing initialization!");

  size_t offset = 0;
  //Cropping laser_reading_ out of fov
  //Assume symmetric offset (i.e., -min_angle = max_angle)
  if (laser_reading_.size() > _config.num_ranges)
    offset = (laser_reading_.size() - _config.num_ranges)/2;

  cloud_.resize(laser_reading_.size()-2*offset);

  int k = 0;
  for (size_t i = offset; i < laser_reading_.size()-offset; ++i){
    const real& r = laser_reading_[i].value;
    
    if (r != r || r <= _config.min_range || r >= _config.max_range)
      continue;

    Vector3 unprojected_point = Vector3::Zero();
    unprojected_point.head<2>() = _table.sincos(i-offset)*r;
    cloud_[k] = LabPoint(laser_reading_[i].label,
                         unprojected_point);
    k++;
  }

  cloud_.resize(k);
    
}
