#!/usr/bin/env python3
import rospy, math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

def normalize_angle(angle):
    """-π..+π 사이로 정규화"""
    return math.atan2(math.sin(angle), math.cos(angle))

class MazeSolver:
    def __init__(self):
        rospy.init_node('maze_solver')
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)
        rospy.Subscriber('/odom', Odometry, self.odom_cb)

        # 속도 파라미터 : Speed Parameters
        self.lin_speed  = 0.2                           # m/s
        self.ang_speed  = 0.7                           # rad/s
        self.avoid_ang_speed = self.ang_speed * 0.2     # rad/s
        # 회전 목표 각도
        self.turn_angle = math.pi/2
        self.u_angle    = math.pi
        # Align 제어 파라미터 : Align control Parameter
        self.kp = 0.3
        self.kd = 0.1

        # Free threshold
        self.min_dist        = 0.43
        self.free_threshold = 0.43      # (m)
        self.front_free_req = 0.75
        self.side_free_req   = 0.8
        self.win_front_deg   = 60
        self.win_side_deg    = 40
        self.avoid_threshold = 0.2      # (m)

        # 상태
        self.state      = 'forward'
        self.err_count = 0
        self.post_turn_dist = None
        self.pos_x = self.pos_y = 0.0

        # 센서 데이터
        self.ranges    = None
        self.angle_inc = None
        # 정면 인덱스 : Front index in 'ranges'
        self.idx_front = 0

        self.rate = rospy.Rate(20)
        # 첫 스캔/odom 대기
        scan = rospy.wait_for_message('/scan', LaserScan)
        self.scan_cb(scan)
        odo  = rospy.wait_for_message('/odom', Odometry)
        self.odom_cb(odo)
        rospy.loginfo("Init complete: scan beams=%d, angle_inc=%.4f",
                      len(self.ranges), self.angle_inc)

    def scan_cb(self, scan: LaserScan):
        self.ranges    = scan.ranges
        self.angle_inc = scan.angle_increment

    def odom_cb(self, odom: Odometry):
        q = odom.pose.pose.orientation
        self.pos_x = odom.pose.pose.position.x
        self.pos_y = odom.pose.pose.position.y

    def count_free_ratio(self, center_idx, win_deg):
        win = int(round(math.radians(win_deg/2) / self.angle_inc))
        n, free, total, inf_count = len(self.ranges), 0, 0, 0
        for di in range(-win, win+1):
            d = self.ranges[(center_idx + di) % n]
            if math.isinf(d):
                inf_count += 1
                continue
            if d > self.min_dist:
                free += 1
            total += 1
        rospy.loginfo("inf value ratio : %.2f", inf_count / total if total > 0 else 0.0)
        return (free/total) if total>0 else 0.0
    
    def distance(self, center_idx, win_deg, metric):
        win = int(round(math.radians(win_deg/2) / self.angle_inc))
        n = len(self.ranges)
        if metric == 'avg':
            # 부채꼴 범위 내에서 유효 거리들의 평균 반환
            # Return average distance of the sector
            total, s = 0, 0.0
            for di in range(-win, win+1):
                d = self.ranges[(center_idx+di) % n]
                if not math.isinf(d):
                    s += d
                    total += 1
            return s/total if total>0 else float('inf')
        
        elif metric == 'med':
            max_dist = 0.0
            for di in range(-win, win+1):
                d = self.ranges[(center_idx+di) % n]
                if not math.isinf(d) and max_dist < d:
                    max_dist = d
            return max_dist
        
        elif metric == 'max':
            max_dist = 0.0
            for di in range(-win, win+1):
                d = self.ranges[(center_idx+di) % n]
                if not math.isinf(d) and max_dist < d:
                    max_dist = d
            return max_dist
        
        elif metric == 'min':
            min_dist = math.inf
            for di in range(-win, win+1):
                d = self.ranges[(center_idx+di) % n]
                if not math.isinf(d) and min_dist > d and d>0.0:
                    min_dist = d
            return min_dist
    
    def align_to_wall(self):
        """가장 가까운 벽과 평행이 되도록 현재 자세를 P(D) 제어로 정렬 - 디버깅용 로그 포함"""
        idx_r = (self.idx_front - int(round(math.pi/2 / self.angle_inc))) % len(self.ranges)
        idx_l = (self.idx_front + int(round(math.pi/2 / self.angle_inc))) % len(self.ranges)
        
        # 정렬용이기 때문에 window size를 작게
        left_dist  = self.distance(idx_l, 5, 'med')
        right_dist = self.distance(idx_r, 5, 'med')

        if left_dist < right_dist:
            idx_center = idx_l
            align_to = 'left'
        else:
            idx_center = idx_r
            align_to = 'right'

        n = len(self.ranges)
        delta_deg = 5
        delta_idx = int(round(math.radians(delta_deg) / self.angle_inc))
        idx_ahead  = (idx_center - delta_idx) % n
        idx_behind = (idx_center + delta_idx) % n

        # 초기 오차·시간 설정
        da = self.ranges[idx_ahead]
        db = self.ranges[idx_behind]
        angle_err = math.atan2(db - da, 2 * math.radians(delta_deg))
        prev_e  = angle_err
        prev_t  = rospy.Time.now().to_sec()
        iter_ct = 0

        rospy.loginfo(">>> align_to_%s_wall START: da=%.3f, db=%.3f, err=%.3f",
                      align_to, da, db, angle_err)

        # 최대 반복 횟수 안전장치
        MAX_ITERS = 1000
        while abs(angle_err) > 0.01 and iter_ct < MAX_ITERS and not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            dt  = max(now - prev_t, 1e-3)

            # 거리 재측정
            da = self.ranges[idx_ahead]
            db = self.ranges[idx_behind]
            angle_err = math.atan2(da - db, 2 * math.radians(delta_deg))
            de = angle_err - prev_e

            # PD 제어
            u = self.kp * angle_err + self.kd * (de / dt)
            u = max(min(u, self.ang_speed), -self.ang_speed)

            if abs(angle_err) < 0.02:
                break

            # 디버깅 로그
            rospy.loginfo("[Align to %s] iter=%02d dt=%.3f da=%.3f db=%.3f err=%.3f de=%.3f u=%.3f",
                          align_to, iter_ct, dt, da, db, angle_err, de, u)

            # 명령 발행
            twist = Twist()
            twist.angular.z = u
            self.cmd_pub.publish(twist)

            # 상태 업데이트
            prev_e  = angle_err
            prev_t  = now
            iter_ct += 1
            self.rate.sleep()

        # 마무리 정지 & 최종 로그
        self.cmd_pub.publish(Twist())
        self.err_count = 0
        rospy.loginfo("<<< align_to_wall END: final err=%.3f after %d iters",
                      angle_err, iter_ct)


    def start_turn(self, kind):
        self.state     = kind
        angle = self.u_angle if kind=='u_turn' else self.turn_angle

        self.turn_duration = abs(angle / self.ang_speed)  # 몇 초 동안 돌아야 하는지
        self.turn_start_time = rospy.Time.now().to_sec()

        if kind == 'turn_right':
            self.turn_dir = -1
        else:
            self.turn_dir = 1

        rospy.loginfo("Start %s: turn_duration=%.2f sec", kind, self.turn_duration)


    def run(self):
        twist = Twist()
        while not rospy.is_shutdown():
            # 0) 회전 후 일정 거리 단순 전진 (앞이 막히면 정지)
            # 0) After the turn, go straight for certain amount
            if self.post_turn_dist is not None:
                p_f = self.count_free_ratio(self.idx_front,  self.win_front_deg)
                dx = self.pos_x - self.start_x
                dy = self.pos_y - self.start_y
                if math.hypot(dx, dy) < self.post_turn_dist and p_f > self.front_free_req:
                    twist.linear.x = self.lin_speed
                    twist.angular.z = 0.0
                    self.cmd_pub.publish(twist)
                    self.rate.sleep()
                    continue
                else:
                    self.post_turn_dist = None

            # 1) Turning
            if self.state != 'forward':
                elapsed = rospy.Time.now().to_sec() - self.turn_start_time
                if elapsed < self.turn_duration:
                    twist.linear.x = 0.0
                    twist.angular.z = self.ang_speed * self.turn_dir
                    self.cmd_pub.publish(twist)
                    rospy.loginfo("Turning... elapsed=%.2f / %.2f", elapsed, self.turn_duration)
                    self.rate.sleep()
                    continue
                else:
                    rospy.loginfo("%s done by timer", self.state)
                    self.cmd_pub.publish(Twist())  # stop
                    self.align_to_wall()
                    self.state = 'forward'
                    self.start_x, self.start_y = self.pos_x, self.pos_y
                    self.post_turn_dist = 0.6
                    continue


            # 2) Forward
            p_f = self.count_free_ratio(self.idx_front,  self.win_front_deg)
            idx_r = (self.idx_front - int(round(math.pi/2 / self.angle_inc))) % len(self.ranges)
            idx_l = (self.idx_front + int(round(math.pi/2 / self.angle_inc))) % len(self.ranges)
            
            p_r = self.count_free_ratio(idx_r, self.win_side_deg)
            p_l = self.count_free_ratio(idx_l, self.win_side_deg)
            left_dist  = self.distance(idx_l, self.win_side_deg, 'med')
            right_dist = self.distance(idx_r, self.win_side_deg, 'med')

            # a) 왼쪽이 비었으면 좌회전 : If left free -> turn left
            if p_l > self.side_free_req:
                self.start_turn('turn_left')
            # b) 앞이 비었다면 직진 : Else if front free -> go straight
            elif p_f > self.front_free_req:
                twist = Twist()

                # 왼쪽이나 오른쪽 벽이 self.threshold이하로 가까워지면 회피
                if left_dist < self.avoid_threshold:
                    print('avoid left wall')
                    twist.angular.z = -self.avoid_ang_speed
                    self.err_count += 1
                elif right_dist < self.avoid_threshold:
                    print('avoid right wall')
                    twist.angular.z = self.avoid_ang_speed
                    self.err_count += 1
                else:
                    print('straight')
                    twist.angular.z = 0.0
                if self.err_count > 20:
                    self.align_to_wall()

                twist.linear.x = self.lin_speed
                self.cmd_pub.publish(twist)
            # c) 오른쪽이 비었다면 우회전 : Else if right free -> turn right
            elif p_r > self.side_free_req:
                self.start_turn('turn_right')
            # d) 좌, 우, 앞 모두 막혔다면 유턴 : If front, left, right blocked -> U turn
            else:
                self.start_turn('u_turn')

            self.rate.sleep()

if __name__=='__main__':
    MazeSolver().run()