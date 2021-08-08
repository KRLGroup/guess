#!/usr/bin/env python
import rospy
import sys
import tf2_ros
import csv
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Twist
from scipy import spatial
import numpy as np


class Landmark:
    def __init__(self, x, y, radius, type):
        self.x = x
        self.y = y
        self.radius = radius
        self.type = type

    def type_index(self):
        if self.type is 'rock':
            return 1
        elif self.type is 'crater':
            return 2
        else:
            return 0

    def __getitem__(self, index):        
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError('Unit coordinates are 2 dimensional')

    def __len__(self):        
        return 2

    def __repr__(self):
        return "Landmark({0}, {1}, {2}, {3})".format(self.x, self.y, self.radius, self.type)


class LandmarkList:
    def __init__(self):
        self.tree = None
        self.list = []
        self.landmark_pub = rospy.Publisher("landmarks", MarkerArray, queue_size=1, latch=True)

    def load_craters(self, filename):
        with open(filename) as f:
            reader = csv.reader(f)
            crater_positions = list(reader)
        for pos in crater_positions:
            self.list.append(Landmark(-(float(pos[1]) - 124.75), -(float(pos[0]) - 124.75), float(pos[2]), 'crater'))

    def load_rocks(self, filename):
        with open(filename) as f:
            reader = csv.reader(f)
            rock_positions = list(reader)
        for pos in rock_positions:
            self.list.append(Landmark(float(pos[1]), -float(pos[0]), float(pos[2]), 'rock'))

    def generate_tree(self):
        self.tree = spatial.KDTree(self.list)

    def query(self, x, y, max_nn, max_radius):
        return [self.list[i] if i != len(self.list) else Landmark(0, 0, 0, 'nil') for i in self.tree.query([x,y], max_nn, distance_upper_bound=max_radius)[1]]

    def publish_markers(self):
        msg = MarkerArray()
        id = 0
        for landmark in self.list:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.id = id
            id += 1
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.scale.x = 2.0 * landmark.radius
            marker.scale.y = 2.0 * landmark.radius
            marker.scale.z = 0.2
            marker.color.a = 0.5
            marker.color.r = 1.0 if landmark.type is 'crater' else 0.0
            marker.color.g = 1.0 if landmark.type is 'rock' else 0.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = landmark.x
            marker.pose.position.y = landmark.y
            msg.markers.append(marker)
        self.landmark_pub.publish(msg)
        print("Published Landmarks")

if __name__ == '__main__':
    rospy.init_node('landmark_markers')
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    landmarks = LandmarkList()
    # landmarks.load_craters('crater_positions.csv')
    landmarks.load_rocks('rock_positions.csv')
    landmarks.generate_tree()
    landmarks.publish_markers()

    min_dist_step = 0.05
    last_pos = None
    dataset = []
    max_radius_rock = 0.0
    max_radius_crater = 0.0
    count = 0
    while not rospy.is_shutdown():
        try:
            trans = tfBuffer.lookup_transform("map", 'base_link', rospy.Time())
            current_pos = [trans.transform.translation.x, trans.transform.translation.y]
            if last_pos is None:
                last_pos = current_pos
            if spatial.distance.euclidean(last_pos, current_pos) >= min_dist_step:
                data = str(rospy.Time.now()) + ' '
                data += str(current_pos[0] - last_pos[0]) + ' '
                data += str(current_pos[1] - last_pos[1]) + ' '
                last_pos = current_pos 
                near_landmarks = landmarks.query(last_pos[0], last_pos[1], 20, 10)
                for l in near_landmarks:
                    if l.type is 'rock' and l.radius > max_radius_rock:
                        max_radius_rock = l.radius
                    elif l.type is 'crater' and l.radius > max_radius_crater:
                        max_radius_crater = l.radius
                    data += '{0} {1} {2} {3} '.format(l.x - current_pos[0], l.y - current_pos[1], l.radius, l.type_index())
                data = data[:-1]
                dataset.append(data)      
                count += 1
                print(count) 
        except:
            continue

    dataset.insert(0, '{0} {1} {2}'.format(len(dataset), max_radius_rock, max_radius_crater))

    with open("landmarks.txt", 'w') as file:
        for row in dataset:
            file.write(str(row) + '\n')
        
    rospy.spin()
