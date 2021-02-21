#! /usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np

from threading import Thread, Lock

import tf
import rospy
import scan_guesser_node.guesser_utils as sgu
import message_filters as mf

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Vector3, PoseWithCovarianceStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray


class GuesserDataAssociator:
    def __init__(self):
        r_id = rospy.get_param('~r_id', 'diag_floor_b1')
        pose_topic = rospy.get_param('~pose_topic', '/amcl_pose')
        scan_topic = rospy.get_param('~scan_topic', '/scan')
        sg_scan_topic = rospy.get_param('~sg_scan_topic', '/sg_scan')
        sv_scan_topic = rospy.get_param('~sv_scan_topic', '/sv_scan')
        sv_latent_topic = rospy.get_param('~sv_latent_topic', '/sv_latent')
        self.scan_sz = int(rospy.get_param('~scan_sz', '512'))

        self.robot_position = np.zeros((1, 2)).astype(np.float32)
        self.p_path, _ = os.path.split(os.path.realpath(__file__))
        self.p_path = self.p_path + "/../../../../dataset/metrics/pplots/" + r_id + "/"
        if not os.path.isdir(self.p_path): os.mkdir(self.p_path)

        self.mtx = Lock()
        self.rate = rospy.Rate(1) # [hz]

        self.pose_sub = mf.Subscriber(pose_topic, PoseWithCovarianceStamped) # , self.poseCb)
        self.scan_sub = mf.Subscriber(scan_topic, LaserScan) # , self.scanCb)
        # self.sg_scan_sub = mf.Subscriber(sg_scan_topic, LaserScan) #, self.scanSgCb)
        self.sv_scan_sub = mf.Subscriber(sv_scan_topic, LaserScan) #, self.scanSvCb)
        self.sv_latent_sub = mf.Subscriber(sv_latent_topic, LaserScan) #, self.latentSvCb)

        self.association_step = 0
        self.max_association_buffer = 0
        self.scan_poses = []
        self.sv_scan_poses = []
        self.association_laser = []
        self.association_latent = []

        self.laser_ass_pub = rospy.Publisher('/laser_association', MarkerArray, queue_size=10)
        self.latent_ass_pub = rospy.Publisher('/latent_association', MarkerArray, queue_size=10)

        ts = mf.ApproximateTimeSynchronizer([
            self.pose_sub,
            self.scan_sub,
            self.sv_scan_sub],
            # self.sv_latent_sub],
                                            10, 0.3, allow_headerless=True)
        ts.registerCallback(self.syncAssociationCb)

        rospy.sleep(2.0)
        sgu.println("-- Spinning ROS node")
        rospy.spin()


    def __insert_scan(self, sample, scan_list, pose, pose_list, thresh=1.0):
        if len(scan_list) == 0:
            scan_list.append(sample)
            pose_list.append([pose])
        else:
            for idx, scan in enumerate(scan_list):
                if self.__l2_distance(scan, sample) < thresh:
                    pose_list[idx].append(pose)
                    return idx

            scan_list.append(sample)
            pose_list.append([pose])

        return -1


    def __l2_distance(self, s1, s2):
        assert s1.shape == s2.shape
        return np.linalg.norm(s1 - s2)


    def __makeMarker(self, mid, x, y, mtype=Marker.SPHERE, scale=Vector3(0.5, 0.5, 0.5),
                     r_ch=0.17647058823, g_ch=0.5294117647, b_ch=0.80392156862):
        m = Marker()
        m.action = Marker.ADD
        m.header.frame_id = '/map'
        m.header.stamp = rospy.Time.now()
        m.ns = 'marker_test_%d' % mtype
        m.id = mid
        m.type = mtype
        m.scale = scale
        m.pose.orientation.y = 0
        m.pose.orientation.w = 1
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.1
        m.color.r = r_ch;
        m.color.g = g_ch;
        m.color.b = b_ch;
        m.color.a = 0.8;
        return m


    def syncAssociationCb(self, pose_data, scan_data, sv_scan_data): # , sv_latent_data):
        r_pose = np.array([pose_data.pose.pose.position.x,
                           pose_data.pose.pose.position.y], dtype=np.float32)
        irange = int(0.5*(len(scan_data.ranges) - self.scan_sz))
        scan = np.array(scan_data.ranges[irange:irange + self.scan_sz], dtype=np.float32)
        sv_scan = np.array(sv_scan_data.ranges, dtype=np.float32)

        print r_pose.shape
        print scan.shape
        print sv_scan.shape


    def scanCb(self, data):
        irange = int(0.5*(len(data.ranges) - self.scan_sz))
        self.mtx.acquire()
        self.scan = data.ranges[irange:irange + self.scan_sz]
        self.mtx.release()


    def scanSgCb(self, data):
        self.mtx.acquire()
        print len(data.ranges)
        self.mtx.release()


    def scanSvCb(self, data):
        self.mtx.acquire()
        print len(data.ranges)
        self.mtx.release()


    def latentSvCb(self, data):
        self.mtx.acquire()
        print len(data.ranges)
        self.mtx.release()


    def __publishAssociation(self, laser_id, latent_id):
        if not laser_id < len(self.scan_poses): return
        if not latent_id < len(self.sv_scan_poses): return

        if False and len(self.scan_poses) > 100:
            self.scan_poses = []
            self.sv_scan_poses = []
            self.association_laser = []
            self.association_latent = []

        laser_poses = self.scan_poses[laser_id]
        sv_scan_poses = self.sv_scan_poses[latent_id]

        marker_array = MarkerArray()
        for ip, pose in enumerate(laser_poses):
            marker_array.markers.append(self.__makeMarker(
                ip, pose[0], pose[1],
                scale=Vector3(0.8, 0.8, 0.8),
                r_ch=0.7, g_ch=0.1, b_ch=0.1))

        print len(marker_array.markers)
        self.laser_ass_pub.publish(marker_array)

        marker_array = MarkerArray()
        for ip, pose in enumerate(sv_scan_poses):
            marker_array.markers.append(self.__makeMarker(
                ip, pose[0], pose[1],
                scale=Vector3(0.8, 0.8, 0.8),
                r_ch=0.3, g_ch=0.0, b_ch=0.9))

        self.latent_ass_pub.publish(marker_array)


    def run(self):
        self.mtx.acquire()
        print "rick"
        self.mtx.release()


        if False and self.association_step % 15 == 0:
            self.association_step = 0
            try:
                r_pos, r_rot = self.tf_listener.lookupTransform(
                    '/map', '/base_laser_link', rospy.Time(0))

                if not isinstance(r_pos, np.ndarray):
                    r_pos = np.array(r_pos, dtype=np.float32)
                if not isinstance(r_rot, np.ndarray):
                    r_rot = np.array(r_rot, dtype=np.float32)

                r_pose = np.concatenate([r_pos, r_rot], axis=-1)

                if not isinstance(data.ranges, np.ndarray):
                    laser_scan = np.array(data.ranges, dtype=np.float32)
                else:
                    laser_scan = data.ranges
                if not isinstance(sg_data[1], np.ndarray):
                    latent_scan = np.array(sg_data[1], dtype=np.float32)
                else:
                    latent_scan = sg_data[1]

                laser_id = self.__insert_scan(laser_scan, self.association_laser,
                                              r_pose.copy(), self.scan_poses, thresh=3.)
                latent_id = self.__insert_scan(latent_scan, self.association_latent,
                                               r_pose.copy(), self.sv_scan_poses, thresh=1.)

                self.__publishAssociation(laser_id, latent_id)

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass


if __name__ == "__main__":
    sgu.println('-- Guesser data associator...')
    rospy.init_node('guesser_data_associator')
    try:
        gda = GuesserDataAssociator()
        # gda.run()
    except rospy.ROSInterruptException: pass
