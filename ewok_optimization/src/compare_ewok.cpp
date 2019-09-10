/**
 * This file is part of Ewok.
 *
 * Copyright 2017 Vladyslav Usenko, Technical University of Munich.
 * Developed by Vladyslav Usenko <vlad dot usenko at tum dot de>,
 * for more information see <http://vision.in.tum.de/research/robotvision/replanning>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * Ewok is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Ewok is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Ewok. If not, see <http://www.gnu.org/licenses/>.
 */

#include <chrono>

#include <ros/ros.h>
#include "std_msgs/Empty.h"

#include <ewok/uniform_bspline_3d_optimization.h>
#include <ewok/polynomial_3d_optimization.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud.h>

pcl::PointCloud<pcl::PointXYZ> latest_cloud;

double start_x, start_y, start_z;
double start_vx, start_vy, start_vz;
double goal_x, goal_y, goal_z;

bool have_goal, have_map;

/* ---------- receive planning map ---------- */
void mapCallback(const sensor_msgs::PointCloud2& msg)
{
  pcl::fromROSMsg(msg, latest_cloud);

  if ((int)latest_cloud.points.size() == 0)
    return;

  have_map = true;
  // std::cout << "[ewok]: get map" << std::endl;
}

void sgCallback(const sensor_msgs::PointCloud& msg)
{
  if (msg.points.size() != 3)
  {
    std::cout << "sg num error." << std::endl;
    return;
  }
  else
  {
    // std::cout << "get sg msg." << std::endl;
  }

  start_x = msg.points[0].x, start_y = msg.points[0].y, start_z = msg.points[0].z;
  start_vx = msg.points[1].x, start_vy = msg.points[1].y, start_vz = msg.points[1].z;
  goal_x = msg.points[2].x, goal_y = msg.points[2].y, goal_z = msg.points[2].z;

  have_goal = true;
  // std::cout << "[ewok]: get sg" << std::endl;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "spline_optimization_example");
  ros::NodeHandle nh;
  ros::NodeHandle node("~");

  ROS_INFO("Started spline_optimization_example");

  ros::Publisher global_traj_pub = nh.advertise<visualization_msgs::MarkerArray>("global_trajectory", 1, true);

  ros::Publisher before_opt_pub = nh.advertise<visualization_msgs::MarkerArray>("before_optimization", 1, true);
  ros::Publisher after_opt_pub = nh.advertise<visualization_msgs::MarkerArray>("after_optimization", 1, true);

  ros::Publisher occ_marker_pub = nh.advertise<visualization_msgs::Marker>("ring_buffer/occupied", 1, true);
  ros::Publisher free_marker_pub = nh.advertise<visualization_msgs::Marker>("ring_buffer/free", 1, true);
  ros::Publisher dist_marker_pub = nh.advertise<visualization_msgs::Marker>("ring_buffer/distance", 1, true);

  // benchmark task
  ros::Subscriber map_sub = node.subscribe("/laser_cloud_surround", 1, mapCallback);
  ros::Subscriber sg_sub = node.subscribe("/start_goal", 1, sgCallback);
  ros::Publisher finish_pub = node.advertise<std_msgs::Empty>("/finish_test", 1, true);

  ros::Duration(1.0).sleep();

  have_goal = false, have_map = false;
  const int use_map_num = 50;
  int exp_num = 0;

  /* ============================== map loop of benchmark ============================== */
  ewok::EuclideanDistanceRingBuffer<9>::Ptr edrb;
  ewok::EuclideanDistanceRingBuffer<9>::PointCloud cloud;

  while (ros::ok())
  {
    /* wait for map and goal ready */
    while (ros::ok())
    {
      if (have_map && have_goal)
        break;

      ros::spinOnce();
    }

    /* ---------- manage map ---------- */
    if (exp_num % use_map_num == 0)
    {
      edrb.reset(new ewok::EuclideanDistanceRingBuffer<9>(0.1, 1));
      cloud.clear();
      for (int i = 0; i < latest_cloud.size(); ++i)
      {
        pcl::PointXYZ pt = latest_cloud.at(i);
        const int step = 1;
        for (int x = -step; x <= step; ++x)
          for (int y = -step; y <= step; ++y)
            for (int z = -step; z <= step; ++z)
            {
              cloud.push_back(Eigen::Vector4f(pt.x + x * 0.1, pt.y + y * 0.1, pt.z + z * 0.1, 0));
            }
      }
      edrb->insertPointCloud(cloud, Eigen::Vector3f(0, 0, 0));
      edrb->insertPointCloud(cloud, Eigen::Vector3f(0, 0, 0));
      edrb->updateDistance();

      // visualization_msgs::Marker m_occ, m_free, m_dist;
      // edrb->getMarkerOccupied(m_occ);
      // occ_marker_pub.publish(m_occ);
    }

    /* ---------- manage start and goal ---------- */
    Eigen::Vector3d start_pt, start_vel, start_acc, end_pt, end_vel;
    start_pt(0) = start_x, start_pt(1) = start_y, start_pt(2) = start_z;
    end_pt(0) = goal_x, end_pt(1) = goal_y, end_pt(2) = goal_z;

    start_vel.setZero();
    start_acc.setZero();
    end_vel.setZero();

    /* ---------- generate global reference trajectory ---------- */
    const Eigen::Vector4d limits(2.0, 1.0, 0, 0), limits2(10.0, 10.0, 0, 0);

    ewok::Polynomial3DOptimization<10> po(limits * 0.8);

    double len = (end_pt - start_pt).norm();
    int seg_num = ceil(len / 2.0);
    seg_num = std::max(2, seg_num);

    typename ewok::Polynomial3DOptimization<10>::Vector3Array vec;
    for (int i = 0; i <= seg_num; ++i)
    {
      Eigen::Vector3d way_pt = double(seg_num - i) / seg_num * start_pt + double(i) / seg_num * end_pt;
      vec.push_back(way_pt);
    }

    auto traj = po.computeTrajectory(vec);

    visualization_msgs::MarkerArray traj_marker;
    traj->getVisualizationMarkerArray(traj_marker, "trajectory", Eigen::Vector3d(1, 0, 0));
    global_traj_pub.publish(traj_marker);

    // std::cout << "[ewok]: duration: " << traj->duration() << std::endl;

    /* ---------- do local replanning ---------- */
    const int num_points = 7;
    const double dt = 0.5;

    Eigen::Vector3d start_point(-5, -5, 0), end_point(5, 5, 0);
    ewok::UniformBSpline3DOptimization<6> spline_opt(traj, dt);

    for (int i = 0; i < num_points; i++)
    {
      spline_opt.addControlPoint(vec[0]);
    }

    spline_opt.setDistanceBuffer(edrb);
    spline_opt.setLimits(limits2);

    // double tc = spline_opt.getClosestTrajectoryTime(Eigen::Vector3d(-3, -5, 1), 2.0);
    // ROS_INFO_STREAM("Closest time: " << tc);
    // ROS_INFO("Finished setting up data");

    double current_time = 0;
    int add_point_num = 0;
    for (double current_time = 0.0; current_time < traj->duration(); current_time += dt)
    {
      double end_time = spline_opt.spline_.maxValidTime() - spline_opt.spline_.minValidTime() - 1e-4;
      Eigen::Vector3d end_mid = traj->evaluate(end_time, 0);
      spline_opt.addControlPoint(end_mid);
      add_point_num += 1;
    }
    spline_opt.setNumControlPointsOptimized(add_point_num);

    visualization_msgs::MarkerArray before_opt_markers, after_opt_markers;
    spline_opt.getMarkers(before_opt_markers, "before_opt", Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(1, 0, 0));

    auto t1 = std::chrono::high_resolution_clock::now();

    double error = spline_opt.optimize();

    auto t2 = std::chrono::high_resolution_clock::now();
    double miliseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1.0e6;
    ROS_INFO_STREAM("Finished optimization in " << miliseconds << " ms. Error: " << error);

    spline_opt.getMarkers(after_opt_markers, "after_opt", Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 1, 1));
    before_opt_pub.publish(before_opt_markers);
    after_opt_pub.publish(after_opt_markers);

    /* ---------- evaluate trajectory ---------- */

    // safety / collision
    int collide = 0;
    for (double ckt = spline_opt.spline_.minValidTime(); ckt < spline_opt.spline_.maxValidTime(); ckt += 0.01)
    {
      Eigen::Vector3d ck_pt = spline_opt.spline_.evaluate(ckt, 0);
      Eigen::Vector3d grad;
      double dist = edrb->getDistanceWithGrad(ck_pt, grad);
      if (dist < 0.1)
      {
        collide = true;
        break;
      }
    }

    // smoothness / energy
    double jerk = 0.0;
    for (double ckt = spline_opt.spline_.minValidTime(); ckt < spline_opt.spline_.maxValidTime(); ckt += 0.01)
    {
      Eigen::Vector3d ck_jk = spline_opt.spline_.evaluate(ckt, 3);
      jerk += ck_jk.norm() * 0.01;
    }

    std::ofstream file("/home/bzhouai/workspaces/plan_ws/src/uav_planning_bm/resource/icra2020boyu/ewok.txt",
                       std::ios::app);
    if (file.fail())
    {
      std::cout << "open file error!\n";
      return -1;
    }

    file << "ewok:" << exp_num + 1 << ",safety:" << collide << ",jerk:" << jerk << "\n";

    file.close();

    /* ---------- finish test flag ---------- */
    std::cout << "[ewok]: finish test." << std::endl;
    ++exp_num;
    have_goal = false;
    if (exp_num % use_map_num == 0)
      have_map = false;

    std_msgs::Empty finish_msg;
    finish_pub.publish(finish_msg);
  }

  ros::spin();

  return 0;
}