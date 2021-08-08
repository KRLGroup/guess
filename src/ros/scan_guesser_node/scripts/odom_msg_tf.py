#!/usr/bin/env python
import rospy
import sys
import tf
from nav_msgs.msg import Odometry

def update_tf(data):
  transform_from = "base_link"
  transform_to   = "odom"
  pos = data.pose.pose.position
  ori = data.pose.pose.orientation
  orientation = [ori.x, ori.y, ori.z, ori.w]
  position    = [pos.x, pos.y, pos.z]
  br = tf.TransformBroadcaster()
  br.sendTransform(position, orientation, rospy.Time.now(), transform_from, transform_to)

if __name__ == '__main__':
  rospy.init_node("odom_msg_tf")
  pose_subscriber = rospy.Subscriber("/odom", Odometry, update_tf)
  rospy.spin()

