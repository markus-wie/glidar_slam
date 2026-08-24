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
  bool csm_debug_enable{false};
  bool csm_debug_images_enable{false};

  // Loop Closure Parameters
  bool loop_debug_enable{false};
  std::size_t loop_input_queue_capacity{2};
  std::size_t loop_output_queue_capacity{8};
  uint64_t loop_minimum_key_separation{20};
  double loop_maximum_distance{2.0};
  double loop_maximum_yaw_difference{1.0};
  double loop_mahalanobis_threshold{3.0};
  double loop_minimum_xy_variance{0.01};
  double loop_minimum_score{0.5};
  double loop_maximum_consistency_error{0.5};

  // iSAM2 Parameters
  double isam_relinearizeThreshold{0.1};
  int isam_relinearizeSkip{10};
};

inline std::ostream & operator<<(std::ostream & os, const Parameters & p)
{
  os << "Parameters{\n"
     << "  odom_frame: " << p.odom_frame << ",\n"
     << "  map_frame: " << p.map_frame << ",\n"
     << "  scan_frame: " << p.scan_frame << ",\n"
     << "  base_frame: " << p.base_frame << ",\n"
     << "  scan_topic: " << p.scan_topic << ",\n"
     << "  tf_topic: " << p.tf_topic << ",\n"
     << "  odom_topic: " << p.odom_topic << ",\n"
     << "  csm_debug_low_topic: " << p.csm_debug_low_topic << ",\n"
     << "  csm_debug_high_topic: " << p.csm_debug_high_topic << ",\n"
     << "  odom_covariance_diagonal: [";

  for (std::size_t i = 0; i < p.odom_covariance_diagonal.size(); ++i) {
    os << p.odom_covariance_diagonal[i] << (i < p.odom_covariance_diagonal.size() - 1 ? ", " : "");
  }

  os << "],\n"
     << "  minimum_travel_distance: " << p.minimum_travel_distance << ",\n"
     << "  minimum_travel_heading: " << p.minimum_travel_heading << ",\n"
     << "  occ_map_resolution: " << p.occ_map_resolution << ",\n"
     << "  occ_map_padding: " << p.occ_map_padding << ",\n"
     << "  submap_window_size: " << p.submap_window_size << ",\n"
     << "  csm_debug_images_enable: " << (p.csm_debug_images_enable ? "true" : "false") << ",\n"
     << "  csm_debug_enable: " << (p.csm_debug_enable ? "true" : "false") << ",\n"
     << "  csm_smear_deviation: " << p.csm_smear_deviation << ",\n"
     << "  csm_use_penalty: " << (p.csm_use_penalty ? "true" : "false") << ",\n"
     << "  csm_distance_variance_penalty: " << p.csm_distance_variance_penalty << ",\n"
     << "  csm_angle_variance_penalty: " << p.csm_angle_variance_penalty << ",\n"
     << "  csm_minimum_distance_penalty: " << p.csm_minimum_distance_penalty << ",\n"
     << "  csm_minimum_angle_penalty: " << p.csm_minimum_angle_penalty << ",\n"
     << "  csm_search_stages (size): " << p.csm_search_stages.size() << ",\n"
     << "  loop_debug_enable: " << (p.loop_debug_enable ? "true" : "false") << ",\n"
     << "  loop_input_queue_capacity: " << p.loop_input_queue_capacity << ",\n"
     << "  loop_output_queue_capacity: " << p.loop_output_queue_capacity << ",\n"
     << "  loop_minimum_key_separation: " << p.loop_minimum_key_separation << ",\n"
     << "  loop_maximum_distance: " << p.loop_maximum_distance << ",\n"
     << "  loop_maximum_yaw_difference: " << p.loop_maximum_yaw_difference << ",\n"
     << "  loop_mahalanobis_threshold: " << p.loop_mahalanobis_threshold << ",\n"
     << "  loop_minimum_xy_variance: " << p.loop_minimum_xy_variance << ",\n"
     << "  loop_minimum_score: " << p.loop_minimum_score << ",\n"
     << "  loop_maximum_consistency_error: " << p.loop_maximum_consistency_error << ",\n"
     << "  isam_relinearizeThreshold: " << p.isam_relinearizeThreshold << ",\n"
     << "  isam_relinearizeSkip: " << p.isam_relinearizeSkip << "\n"
     << "}";

  return os;
}

}  // namespace glidar_slam::core
