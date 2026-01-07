#!/usr/bin/env python
import numpy as np
import argparse
import rospy
from geometry_msgs.msg import WrenchStamped
from omni_msgs.msg import OmniState
from skrobot.models import Nextage
from skrobot.interfaces.ros import NextageROSRobotInterface
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import quaternion2matrix
from skrobot.viewers import TrimeshSceneViewer
from skrobot.model import Axis, Box
import threading

class NextageTeleop:
    def __init__(self, use_viewer=True, show_angles=False):
        rospy.init_node('nextage_teleop')
        
        self.robot = Nextage()
        self.ri = NextageROSRobotInterface(self.robot)
        self.show_angles = show_angles
        
        self.base_offset_l = np.array([300, 100, -60])
        self.base_offset_r = np.array([300, -100, -60])
        self.ratio = 0.7
        self.send_time = 200
        
        self.end_coords_offset = np.array([-0.005, 0.0, 0.01])
        
        self.R_map = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        
        self.R_yaw_flip = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        
        self.collision_ignore_pairs = {
            frozenset(['CHEST_JOINT0_Link', 'HEAD_JOINT1_Link']),
            frozenset(['LARM_JOINT4_Link', 'LARM_JOINT5_Link']),
            frozenset(['RARM_JOINT4_Link', 'RARM_JOINT5_Link']),
            frozenset(['CHEST_JOINT0_Link', 'WAIST']),
            frozenset(['CHEST_JOINT0_Link', 'HEAD_JOINT0_Link']),
            frozenset(['HEAD_JOINT0_Link', 'HEAD_JOINT1_Link']),
            frozenset(['CHEST_JOINT0_Link', 'LARM_JOINT0_Link']),
            frozenset(['LARM_JOINT0_Link', 'LARM_JOINT1_Link']),
            frozenset(['LARM_JOINT1_Link', 'LARM_JOINT2_Link']),
            frozenset(['LARM_JOINT2_Link', 'LARM_JOINT3_Link']),
            frozenset(['LARM_JOINT3_Link', 'LARM_JOINT4_Link']),
            frozenset(['CHEST_JOINT0_Link', 'RARM_JOINT0_Link']),
            frozenset(['RARM_JOINT0_Link', 'RARM_JOINT1_Link']),
            frozenset(['RARM_JOINT1_Link', 'RARM_JOINT2_Link']),
            frozenset(['RARM_JOINT2_Link', 'RARM_JOINT3_Link']),
            frozenset(['RARM_JOINT3_Link', 'RARM_JOINT4_Link']),
            frozenset(['LARM_JOINT5_Link', 'RARM_JOINT5_Link']),
        }
        
        self.lpos = None
        self.lrot = None
        self.lclose = None
        self.llocked = None
        self.rpos = None
        self.rrot = None
        self.rclose = None
        self.rlocked = None
        
        self.prev_llock = None
        self.prev_rlock = None
        
        self.xforce_offset = 0
        self.yforce_offset = 0
        self.zforce_offset = 0
        self.rforce_calibrated = False
        
        rospy.Subscriber('/robotiq_ft_wrench', WrenchStamped, self.rforce_cb)
        rospy.Subscriber('/left_device/phantom/state', OmniState, self.left_phantom_cb)
        rospy.Subscriber('/right_device/phantom/state', OmniState, self.right_phantom_cb)
        
        self.robot.reset_pose()
        self.robot.head.angle_vector([0, np.deg2rad(60)])
        self.robot.larm.move_end_pos([0, 0, -0.1])
        self.robot.rarm.move_end_pos([0, 0, -0.1])
        av = self.robot.angle_vector()
        self.ri.angle_vector(av, time=1.0)
        self.ri.wait_interpolation()
        
        self.use_viewer = use_viewer
        if self.use_viewer:
            self.viewer = TrimeshSceneViewer(resolution=(800, 600))
            self.viewer.add(self.robot)
            
            self.lc = Box(extents=[0.01, 0.01, 0.01], pos=self.base_offset_l/1000.0)
            self.lc.visual_mesh.visual.face_colors = [255, 255, 0, 255]
            self.viewer.add(self.lc)
            
            self.rc = Box(extents=[0.01, 0.01, 0.01], pos=self.base_offset_r/1000.0)
            self.rc.visual_mesh.visual.face_colors = [255, 255, 0, 255]
            self.viewer.add(self.rc)
            
            self.la_target = Axis(axis_radius=0.003, axis_length=0.03)
            self.viewer.add(self.la_target)
            
            self.ra_target = Axis(axis_radius=0.003, axis_length=0.03)
            self.viewer.add(self.ra_target)
            
            self.la_end = Axis(axis_radius=0.003, axis_length=0.03)
            self.viewer.add(self.la_end)
            
            self.ra_end = Axis(axis_radius=0.003, axis_length=0.03)
            self.viewer.add(self.ra_end)
            
            self.viewer.show()
            self.viewer.set_camera([np.deg2rad(45), -np.deg2rad(0), np.deg2rad(135)], distance=1.5)
            
            self.viewer_thread = threading.Thread(target=self.viewer_loop)
            self.viewer_thread.daemon = True
            self.viewer_thread.start()
        
        rospy.sleep(1)
        
    def viewer_loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown() and self.viewer.is_active:
            if self.use_viewer:
                self.la_end.newcoords(self.robot.larm.end_coords.copy_worldcoords())
                self.ra_end.newcoords(self.robot.rarm.end_coords.copy_worldcoords())
            self.viewer.redraw()
            rate.sleep()
        
    def left_phantom_cb(self, msg):
        self.lpos = msg.pose.position
        self.lrot = msg.pose.orientation
        self.lclose = msg.close_gripper
        self.llocked = msg.locked
        
    def right_phantom_cb(self, msg):
        self.rpos = msg.pose.position
        self.rrot = msg.pose.orientation
        self.rclose = msg.close_gripper
        self.rlocked = msg.locked
        
    def rforce_cb(self, msg):
        if not self.rforce_calibrated:
            self.xforce_offset = msg.wrench.force.x
            self.yforce_offset = msg.wrench.force.y
            self.zforce_offset = msg.wrench.force.z
            self.rforce_calibrated = True
        
    def check_work_limit(self):
        lh = self.robot.larm.end_coords.worldpos()[2]
        rh = self.robot.rarm.end_coords.worldpos()[2]
        return lh > -0.125 and rh > -0.125
    
    def filtered_self_collision_check(self):
        is_collision, collision_pairs = self.robot.self_collision_check()
        
        if not is_collision:
            return False
        
        filtered_pairs = set()
        for pair in collision_pairs:
            pair_set = frozenset(pair)
            if pair_set not in self.collision_ignore_pairs:
                filtered_pairs.add(pair)
        
        if len(filtered_pairs) > 0:
            rospy.logwarn_throttle(5.0, "Collision pairs: {}".format(filtered_pairs))
            return True
        
        return False
    
    def print_current_angles(self):
        av = self.robot.angle_vector()
        print("\nCurrent joint angles (rad):")
        print(repr(av))
        print("\nCurrent joint angles (deg):")
        print(repr(np.degrees(av)))
        
    def run(self):
        rate = rospy.Rate(1000.0 / self.send_time)
        angle_print_counter = 0
        angle_print_interval = int(1000.0 / self.send_time)
        
        while not rospy.is_shutdown():
            ik_success = True
            
            if self.lpos and self.lrot and not self.llocked:
                lx = (-self.lpos.y * self.ratio + self.base_offset_l[0]) / 1000.0
                ly = (self.lpos.x * self.ratio + self.base_offset_l[1]) / 1000.0
                lz = (self.lpos.z * self.ratio + self.base_offset_l[2]) / 1000.0
                
                lx += self.end_coords_offset[0]
                ly += self.end_coords_offset[1]
                lz += self.end_coords_offset[2]
                
                lR_src = quaternion2matrix([self.lrot.w, self.lrot.x, self.lrot.y, self.lrot.z])
                lR_dst = self.R_map @ lR_src
                lR_dst = lR_dst @ self.R_yaw_flip
                
                larm_target = Coordinates(pos=[lx, ly, lz], rot=lR_dst)
                larm_target.rotate(np.pi, 'y')
                larm_target.rotate(np.pi, 'z')
                
                if self.use_viewer:
                    self.la_target.newcoords(larm_target)
                
                result = self.robot.larm.inverse_kinematics(
                    larm_target, 
                    rotation_axis=True,
                    stop=10,
                    revert_if_fail=True
                )
                if result is False:
                    ik_success = False
                    
            if self.rpos and self.rrot and not self.rlocked:
                rx = (-self.rpos.y * self.ratio + self.base_offset_r[0]) / 1000.0
                ry = (self.rpos.x * self.ratio + self.base_offset_r[1]) / 1000.0
                rz = (self.rpos.z * self.ratio + self.base_offset_r[2]) / 1000.0
                
                rx += self.end_coords_offset[0]
                ry += self.end_coords_offset[1]
                rz += self.end_coords_offset[2]
                
                rR_src = quaternion2matrix([self.rrot.w, self.rrot.x, self.rrot.y, self.rrot.z])
                rR_dst = self.R_map @ rR_src
                rR_dst = rR_dst @ self.R_yaw_flip
                
                rarm_target = Coordinates(pos=[rx, ry, rz], rot=rR_dst)
                rarm_target.rotate(np.pi, 'y')
                rarm_target.rotate(np.pi, 'z')
                
                if self.use_viewer:
                    self.ra_target.newcoords(rarm_target)
                
                result = self.robot.rarm.inverse_kinematics(
                    rarm_target,
                    rotation_axis=True,
                    stop=10,
                    revert_if_fail=True
                )
                if result is False:
                    ik_success = False
                    
            if self.lclose != self.prev_llock and self.lclose is not None:
                if self.lclose:
                    self.ri.open_forceps('larm')
                else:
                    self.ri.close_forceps('larm')
                self.prev_llock = self.lclose
                
            if self.rclose != self.prev_rlock and self.rclose is not None:
                if self.rclose:
                    self.ri.open_holder('rarm')
                else:
                    self.ri.close_holder('rarm')
                self.prev_rlock = self.rclose
            
            collision_check = self.filtered_self_collision_check()
            work_limit_check = self.check_work_limit()
            
            if ik_success and not collision_check and work_limit_check:
                self.ri.angle_vector(self.robot.angle_vector(), time=self.send_time/1000.0)
            
            if self.show_angles:
                angle_print_counter += 1
                if angle_print_counter >= angle_print_interval:
                    self.print_current_angles()
                    angle_print_counter = 0
                
            rate.sleep()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-viewer', action='store_true')
    parser.add_argument('--show-angles', action='store_true', help='Display current joint angles every second')
    args = parser.parse_args()
    
    teleop = NextageTeleop(use_viewer=not args.no_viewer, show_angles=args.show_angles)
    try:
        teleop.run()
    except KeyboardInterrupt:
        if teleop.use_viewer:
            teleop.viewer.close()
