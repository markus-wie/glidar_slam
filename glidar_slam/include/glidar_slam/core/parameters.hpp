#pragma once

#include <string>
#include <vector>

namespace glidar_slam::core {

struct CsmSearchStage
{
  double field_resolution;
  double translation_step;
  double angular_step;
  double window_x;
  double window_y;
  double window_yaw;
};

struct Parameters
{
  // Coordinate Frame Parameters
  std::string odom_frame{"odom"};
  std::string map_frame{"map"};
  std::string scan_frame{"laser"};
  std::string base_frame{"base_link"};

  // ROS Topic Parameters
  std::string scan_topic{"/scan"};
  std::string tf_topic{"/tf"};
  std::string odom_topic{"/odometry/filtered"};
  std::string csm_debug_low_topic{"/glidar_slam/debug/csm_likelihood_low"};
  std::string csm_debug_high_topic{"/glidar_slam/debug/csm_likelihood_high"};

  std::vector<double> odom_covariance_diagonal{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0};

  // General Parameters
  double minimum_travel_distance{0.5};
  double minimum_travel_heading{0.5};
  bool enable_csm_debug_images{false};

  // Occupancy Grid Parameters
  double occ_map_resolution{0.05};
  int occ_map_padding{2};

  // Submap Parameters
  int submap_window_size{5};

  // Correlative Scan Matcher Parameters
  double csm_smear_deviation{0.1};
  bool csm_use_penalty{true};
  double csm_distance_variance_penalty{0.5};
  double csm_angle_variance_penalty{1.0};
  double csm_minimum_distance_penalty{0.5};
  double csm_minimum_angle_penalty{0.9};
  std::vector<CsmSearchStage> csm_search_stages;

  // iSAM2 Parameters
  double isam_relinearizeThreshold{0.1};
  int isam_relinearizeSkip{10};
};

}  // namespace glidar_slam::core
