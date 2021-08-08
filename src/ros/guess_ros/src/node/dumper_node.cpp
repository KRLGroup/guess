// ROS
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/LaserScan.h>

#include <fstream>

using namespace std;

//bdc last cmd_vel message
// since the cmd_vel doesn't have a timestamp, we cannot use approximateTime synchro
// ...so this is ugly hand made
geometry_msgs::Twist last_cmd_vel;

//bdc ugly variables
bool got_first_cmd_vel;
bool got_first_time;
ros::Time first_time;

//bdc file writer
std::ofstream *writer;

//bdc Config for storing user parameters
struct Config{
  std::string cmd_vel_topic;
  std::string scan_topic;
  std::string dataset_name;
  float range_min;
  Config() {
    cmd_vel_topic = "/cmd_vel";
    scan_topic    = "/scan_simulated_throttle";
    dataset_name  = "output_data.txt";
    range_min     = 0.1;
  }
} config;

//bdc callback cmd_vel
void callbackCmdVel(const geometry_msgs::Twist::ConstPtr& cmd_vel) {
  last_cmd_vel = *cmd_vel;
  got_first_cmd_vel = true;
}

//bdc callback scan
void callbackScan(const sensor_msgs::LaserScan::ConstPtr& scan);


int main(int argc, char **argv) {

  ros::init(argc, argv, "dumper_node");
  ros::NodeHandle nh("~");

  //bdc init stuff
  got_first_cmd_vel = false;
  got_first_time    = false;
  writer = new std::ofstream(config.dataset_name);

  
  //bdc setup subscribers
  ros::Subscriber cmd_vel_sub = nh.subscribe(config.cmd_vel_topic, 100, callbackCmdVel);
  ros::Subscriber scan_sub = nh.subscribe(config.scan_topic, 100, callbackScan);

  ros::spin();

  writer->close();
}


void callbackScan(const sensor_msgs::LaserScan::ConstPtr& scan) {
  if(!got_first_cmd_vel)
    return;
  const ros::Time& current_scan_time = scan->header.stamp;
  if(!got_first_time) {
    first_time = current_scan_time;
    got_first_time = true;
  }

  //bdc write timestamp (in human readable format)
  const double current_time = (current_scan_time - first_time).toSec();
  *writer << current_time << " ";

  //bdc write last cmd_vel
  *writer << last_cmd_vel.linear.x << " "
          << last_cmd_vel.linear.y << " "
          << last_cmd_vel.linear.z << " "
          << last_cmd_vel.angular.x << " "
          << last_cmd_vel.angular.y << " "
          << last_cmd_vel.angular.z << " ";
  
  //bdc write current scan  
  const std::vector<float>& ranges = scan->ranges;
  for(const float& range : ranges) {
    if(range < config.range_min || std::isnan(range))
      *writer << 0 << " ";
    else
      *writer << range << " ";
  }
  *writer << std::endl;
    
  std::cerr << "x";
}
