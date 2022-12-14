#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from geometry_msgs.msg  import Twist, Point, Pose2D
from sensor_msgs.msg  import LaserScan, JointState
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from math import cos,sin
from gazebo_msgs.srv import GetModelState, GetLinkState
from gazebo_msgs.msg import ModelState, LinkState, LinkStates
import numpy as np
import message_filters
import math
import time

pub = rospy.Publisher('/odometry_p3dx', Pose2D, queue_size = 15)
pub_gt = rospy.Publisher('/ground_truth', Pose2D, queue_size = 15)
pub_ekf = rospy.Publisher('/ekf_p3dx', Pose2D, queue_size = 15)

time = 0
previous_left_angular_position = 0 # MODELO PRECISA PARTIR DO 0!
previous_right_angular_position = 0 # MODELO PRECISA PARTIR DO 0!
x = 0 # MODELO PRECISA PARTIR DO 0!
y = 0 # MODELO PRECISA PARTIR DO 0!
theta = 0 # MODELO PRECISA PARTIR DO 0!

r = 0.179/2 # raio da roda
L = 0.314/2 # largura do robô
index = 0


W2R = r*np.array([[1/2, 1/2], [1/(2*L), -1/(2*L)]])


world_map = [[2, 0, 10, 2.144], 
[10, 2.144, 10, 8.144],
 [10, 8.144, 8.144, 10], 
 [8.144, 10, 1, 10],
 [1, 10, 1, 6], 
 [1, 6, 0, 6], 
 [0, 6, 0, 2], 
 [0, 2, 2, 2], 
 [2, 2, 2, 0]]
angles = np.arange(-90, 91, 9)

kr = 0.1 # constante de deslizamento das rodas
kl = 0.1 # constante de deslizamento das rodas
l = 0.314/2 # distância entra a roda e o centro do robô


P_minus = 0
P_plus = 0
x_minus = (5, 5, 0)

size_matrix = 500
distances = np.zeros((21, size_matrix))

index_W = 0

W = np.zeros((21, 21))
np.fill_diagonal(W, [0.00010139719856420102,
0.00010375093446012301,
0.00010944264232882114,
0.00010713490738979452,
9.098573716983078e-05,
0.0001018220137118724,
9.125890133335905e-05,
0.0001049158329427882,
9.507503969805381e-05,
9.917482196406041e-05,
0.00010108830709550513,
0.0001023556708252036,
9.752547674295316e-05,
0.00010539842282272365,
0.00010420892521968556,
0.00010383731492720378,
9.843542015824919e-05,
0.00010790971018186624,
9.349423762073554e-05,
0.00010208039484222905,
9.200306762192762e-05])

def get_gazebo_link_state(link_name):
    g_get_state = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
    rospy.wait_for_service("/gazebo/get_link_state")
    try:
        state = g_get_state(link_name=link_name)
        now = rospy.get_rostime()
    except Exception(e):
        rospy.logerr('Error on calling service: %s',str(e))
        return
    return state, now

def update_ekf(msg1, msg2):
    global time, previous_left_angular_position, previous_right_angular_position, x, y, theta, index, index_W, distances, size_matrix, W, P_plus

    # obtendo a diferença de intervalo e as leituras de distância

    now = rospy.get_rostime()
    delta_T = now.to_sec() - time
    ranges = msg2.ranges
    if (delta_T > 0):
        # obtendo as velocidades com base no tópico de odometria

        x = msg1.pose.pose.position.x
        y = msg1.pose.pose.position.y
        
        theta = np.degrees(get_rotation(msg1.pose.pose.orientation))

        x_minus = (x, y, theta)

        vel_x = msg1.twist.twist.linear.x
        vel_y = msg1.twist.twist.linear.y
        vel_linear = np.sqrt(math.pow(vel_x, 2) + math.pow(vel_y, 2))
        vel_angular = msg1.twist.twist.angular.z

        ds = vel_linear*delta_T
        dth = vel_angular*delta_T
        dsr = ds+l*dth
        dsl = ds-l*dth

        # incerteza V do deslocamento das rodas
        V_matrix = np.array([[kr * np.abs(dsr), 0], [0, kl*np.abs(dsl)]])

        # Jacobianos das funções P e V, do modelo dinâmico do robô
        Fp = np.matrix([[1, 0, -ds*np.sin(theta + (dth/2))], [0, 1, ds*np.cos(theta + (dth/2))], [0, 0, 1]])

        Fv = np.matrix([[0.5 * np.cos(theta + (dth/2)) - (ds / (4*l)) * np.sin(theta + (dth/2)),
        0.5 * np.cos(theta + (dth/2)) + (ds / (4*l)) * np.sin(theta + (dth/2))],
         [0.5 * np.sin(theta + (dth/2)) + (ds / (4*l)) * np.cos(theta + (dth/2)),
          0.5 * np.sin(theta + (dth/2)) - (ds / (4*l)) * np.cos(theta + (dth/2))], [1/(2*l), 1/(2*l)]])
        
        if (index == 0) :
            P_minus = np.matmul(np.matmul(Fv, V_matrix), Fv.T)
        else:
            P_minus = np.matmul(np.matmul(Fp, P_plus), Fp.T) + np.matmul(np.matmul(Fv, V_matrix), Fv.T)
        
        # Sensor Laser Virtual
        Dist = np.zeros(angles.shape)


        distances = []

        for i in range(0, angles.shape[0]):
            Dist[i] = 9999

        for i in range(0, angles.shape[0]):
            x1 = x
            y1 = y
            angle = angles[i]
            x2 = x1 + np.cos(angle)
            y2 = y1 + np.sin(angle)
            a = y1 - y2
            b = x2 - x1
            c = (x1*y2-x2*y1)*-1
            for wall in world_map:
                x3 = wall[0]
                y3 = wall[1]
                x4 = wall[2]
                y4 = wall[3]

                den = (x1-x2) * (y3-y4)-(y1-y2)*(x3-x4)
                if (den != 0): # caso não seja paralelo
                    Px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/den
                    Py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/den
                    arc_tan = math.atan2(Py-y, Px-x)
                    arc_tan_degrees = np.degrees(arc_tan)
                    thP = np.mod(arc_tan_degrees - theta, 360)
                    if (thP > 180):
                        thP = thP - 360
                    epsilon = 0.1
                    if ((np.min([x3, x4]) - epsilon <= Px)
                    and (Px <= np.max([x3, x4]) + epsilon)
                    and (np.min([y3, y4]) - epsilon <= Py)
                    and (Py <= np.max([y3, y4]) + epsilon)):
                        distances.append(np.sqrt(math.pow(x - Px, 2) + math.pow(y - Py, 2)))
            if (len(distances) > 0):
                Dist[i] = np.min(distances)
                distances = []
        innovation = ranges - Dist


        H = np.zeros((angles.shape[0], 3))

        for i in range(0, innovation.shape[0]):
            x1 = x_minus[0] + ranges[i]*np.cos(x_minus[2] + angles[i])
            y1 = x_minus[1] + ranges[i]*np.sin(x_minus[2] + angles[i])
            H[i][0] = -(x1-x_minus[0]) / np.sqrt((math.pow(x1 - x_minus[0], 2) + math.pow(y1 - x_minus[1], 2)))
            H[i][1] = -(y1-x_minus[1]) / np.sqrt((math.pow(x1 - x_minus[0], 2) + math.pow(y1 - x_minus[1], 2)))


        # Ganho de Kalman
        PkHT = np.matmul(P_minus, H.T)

        HPkHT_W = np.matmul(np.matmul(H, P_minus), H.T) + W

        K = np.matmul(PkHT, np.linalg.inv(HPkHT_W))

        P_plus = P_minus - np.matmul(np.matmul(K, H), P_minus)
        x_plus = x_minus + np.matmul(K, innovation)

        print(K)

        # publicando a odometria

        pub.publish(Pose2D(x, y, theta))

        # republicando o ground truth
        gt = get_ground_truth()
        pub_gt.publish(Pose2D(gt[0], gt[1], np.degrees(gt[2])))
        index = index + 1

        # publicando o EKF
        pub_ekf.publish(Pose2D(x_plus.item(0, 0), x_plus.item(0, 1), x_plus.item(0, 2)))
        

    
    

def ekf():
    rospy.init_node('odometry', anonymous=True)

    laser_sub = message_filters.Subscriber('/p3dx/laser/scan', LaserScan)
    joint_sub = message_filters.Subscriber("/p3dx/base_pose_ground_truth", Odometry)

    ts = message_filters.ApproximateTimeSynchronizer([joint_sub, laser_sub], 1, 0.1)
    ts.registerCallback(update_ekf)

    time = rospy.get_rostime().to_sec()
    while not rospy.is_shutdown():
        a = 1

def get_ground_truth():
    g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    rospy.wait_for_service("/gazebo/get_model_state")
    try:
        state = g_get_state(model_name="p3dx")
    except Exception(e):
        rospy.logerr('Error on calling service: %s',str(e))
        return
    x_gt = state.pose.position.x
    y_gt = state.pose.position.y 
    yaw_gt = get_rotation(state.pose.orientation)
    return [x_gt, y_gt, yaw_gt]


def convert_euler(msg):
    orientation_q = msg
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    return euler_from_quaternion (orientation_list)

def get_rotation (msg):
    roll, pitch, yaw = convert_euler(msg)
    return (yaw)

if __name__ == '__main__':
    ekf()
