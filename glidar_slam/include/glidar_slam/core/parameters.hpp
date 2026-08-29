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
  std::string camera_frame{"camera"};
  std::string base_frame{"base_link"};

  // ROS Topic Parameters
  std::string scan_topic{"/scan"};
  std::string tf_topic{"/tf"};
  std::string odom_topic{"/odometry/filtered"};

  // ROS Topic Parameters (Subscribers)
  std::string color_image_topic{"/camera/color/image_raw"};
  std::string aligned_depth_image_topic{"/camera/aligned_depth/image_raw"};
  std::string color_camera_info_topic{"/camera/color/camera_info"};

  // Synchronization Parameter
  double approx_sync_max_interval{0.1};

  // Odometry Covariance Override [x, y, z, roll, pitch, yaw]
  std::vector<double> odom_covariance_diagonal{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0};

  // General Parameters
  double minimum_travel_distance{0.5};
  double minimum_travel_heading{0.5};
  double unobservable_variance{1e6};
  int submap_window_size{5};

  // Localization Parameters
  bool localization_mode{false};
  double localization_minimum_score{0.5};

  // Debugging Parameters
  bool debug_visualize_covariances{false};
  bool debug_timings{false};
  bool ground_debug_enable{false};
  bool ground_matching_debug_enable{false};
  bool loop_debug_enable{false};
  bool csm_debug_enable{false};

  // LiDAR Preprocessing Parameters
  bool scan_voxelization_enable{true};
  double scan_voxelization_size{0.2};
  bool scan_densification_enable{true};

  // Mapping Parameters
  double mapping_occupancy_resolution{0.05};
  double mapping_ground_resolution{0.02};
  bool mapping_ground_enable_texture_mapping{true};
  bool mapping_ground_enable_ground_marking_mapping{false};
  int mapping_threshold{200};
  double mapping_log_odds_hit{0.8};
  double mapping_log_odds_miss{0.35};
  double mapping_log_odds_cap{15.0};

  // Ground Factor
  bool ground_optimization_enable{true};
  double ground_normal_sigma{0.05};
  double ground_distance_sigma{0.03};

  // Ground Extraction Parameters
  std::vector<float> ground_roi_ratios{0.0F, 0.0F, 0.2F, 0.4F};
  int ground_extraction_pixel_stride{2};
  double ground_extraction_voxel_size{0.05};
  double ground_extraction_max_distance{5.0};
  double ground_extraction_distance_threshold{0.05};
  int ground_extraction_adaptive_threshold_block_size{101};
  double ground_extraction_adaptive_threshold_C{-65.0};

  // Ground Matching Parameters
  bool ground_matching_enable{false};
  double ground_matching_minimum_score{0.5};
  double ground_matching_max_distance{2.0};

  // Loop Closure Parameters
  uint64_t loop_minimum_key_separation{20};
  double loop_maximum_yaw_difference{1.0};
  double loop_mahalanobis_threshold{3.0};
  double loop_minimum_score{0.5};

  // Correlative Scan Matcher Parameters
  double csm_smear_deviation{0.1};
  bool csm_use_laplace_kernel{false};
  bool csm_use_distance_transform{false};
  bool csm_use_tbb{false};
  std::vector<CsmSearchStage> csm_search_stages;

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
     << "  camera_frame: " << p.camera_frame << ",\n"
     << "  base_frame: " << p.base_frame << ",\n"
     << "  scan_topic: " << p.scan_topic << ",\n"
     << "  tf_topic: " << p.tf_topic << ",\n"
     << "  odom_topic: " << p.odom_topic << ",\n"
     << "  color_image_topic: " << p.color_image_topic << ",\n"
     << "  aligned_depth_image_topic: " << p.aligned_depth_image_topic << ",\n"
     << "  color_camera_info_topic: " << p.color_camera_info_topic << ",\n"
     << "  approx_sync_max_interval: " << p.approx_sync_max_interval << ",\n"
     << "  odom_covariance_diagonal: [";

  for (std::size_t i = 0; i < p.odom_covariance_diagonal.size(); ++i) {
    os << p.odom_covariance_diagonal[i] << (i < p.odom_covariance_diagonal.size() - 1 ? ", " : "");
  }

  os << "],\n"
     << "  minimum_travel_distance: " << p.minimum_travel_distance << ",\n"
     << "  minimum_travel_heading: " << p.minimum_travel_heading << ",\n"
     << "  unobservable_variance: " << p.unobservable_variance << ",\n"
     << "  submap_window_size: " << p.submap_window_size << ",\n"
     << "  localization_mode: " << (p.localization_mode ? "true" : "false") << ",\n"
     << "  localization_minimum_score: " << p.localization_minimum_score << ",\n"
     << "  debug_visualize_covariances: " << (p.debug_visualize_covariances ? "true" : "false")
     << ",\n"
     << "  debug_timings: " << (p.debug_timings ? "true" : "false") << ",\n"
     << "  ground_debug_enable: " << (p.ground_debug_enable ? "true" : "false") << ",\n"
     << "  ground_matching_debug_enable: " << (p.ground_matching_debug_enable ? "true" : "false")
     << ",\n"
     << "  loop_debug_enable: " << (p.loop_debug_enable ? "true" : "false") << ",\n"
     << "  csm_debug_enable: " << (p.csm_debug_enable ? "true" : "false") << ",\n"
     << "  scan_voxelization_enable: " << (p.scan_voxelization_enable ? "true" : "false") << ",\n"
     << "  scan_voxelization_size: " << p.scan_voxelization_size << ",\n"
     << "  scan_densification_enable: " << (p.scan_densification_enable ? "true" : "false") << ",\n"
     << "  mapping_occupancy_resolution: " << p.mapping_occupancy_resolution << ",\n"
     << "  mapping_ground_resolution: " << p.mapping_ground_resolution << ",\n"
     << "  mapping_ground_enable_texture_mapping: "
     << (p.mapping_ground_enable_texture_mapping ? "true" : "false") << ",\n"
     << "  mapping_ground_enable_ground_marking_mapping: "
     << (p.mapping_ground_enable_ground_marking_mapping ? "true" : "false") << ",\n"
     << "  mapping_threshold: " << p.mapping_threshold << ",\n"
     << "  mapping_log_odds_hit: " << p.mapping_log_odds_hit << ",\n"
     << "  mapping_log_odds_miss: " << p.mapping_log_odds_miss << ",\n"
     << "  mapping_log_odds_cap: " << p.mapping_log_odds_cap << ",\n"
     << "  ground_optimization_enable: " << (p.ground_optimization_enable ? "true" : "false")
     << ",\n"
     << "  ground_normal_sigma: " << p.ground_normal_sigma << ",\n"
     << "  ground_distance_sigma: " << p.ground_distance_sigma << ",\n"
     << "  ground_roi_ratios: [";

  for (std::size_t i = 0; i < p.ground_roi_ratios.size(); ++i) {
    os << p.ground_roi_ratios[i] << (i < p.ground_roi_ratios.size() - 1 ? ", " : "");
  }

  os << "],\n"
     << "  ground_extraction_pixel_stride: " << p.ground_extraction_pixel_stride << ",\n"
     << "  ground_extraction_voxel_size: " << p.ground_extraction_voxel_size << ",\n"
     << "  ground_extraction_max_distance: " << p.ground_extraction_max_distance << ",\n"
     << "  ground_extraction_distance_threshold: " << p.ground_extraction_distance_threshold
     << ",\n"
     << "  ground_extraction_adaptive_threshold_block_size: "
     << p.ground_extraction_adaptive_threshold_block_size << ",\n"
     << "  ground_extraction_adaptive_threshold_C: " << p.ground_extraction_adaptive_threshold_C
     << ",\n"
     << "  ground_matching_enable: " << (p.ground_matching_enable ? "true" : "false") << ",\n"
     << "  ground_matching_minimum_score: " << p.ground_matching_minimum_score << ",\n"
     << "  ground_matching_max_distance: " << p.ground_matching_max_distance << ",\n"
     << "  loop_minimum_key_separation: " << p.loop_minimum_key_separation << ",\n"
     << "  loop_maximum_yaw_difference: " << p.loop_maximum_yaw_difference << ",\n"
     << "  loop_mahalanobis_threshold: " << p.loop_mahalanobis_threshold << ",\n"
     << "  loop_minimum_score: " << p.loop_minimum_score << ",\n"
     << "  csm_smear_deviation: " << p.csm_smear_deviation << ",\n"
     << "  csm_use_laplace_kernel: " << (p.csm_use_laplace_kernel ? "true" : "false") << ",\n"
     << "  csm_use_distance_transform: " << (p.csm_use_distance_transform ? "true" : "false")
     << ",\n"
     << "  csm_use_tbb: " << (p.csm_use_tbb ? "true" : "false") << ",\n"
     << "  csm_search_stages: [\n";

  for (std::size_t i = 0; i < p.csm_search_stages.size(); ++i) {
    const auto & s = p.csm_search_stages[i];
    os << "    {field_resolution: " << s.field_resolution
       << ", translation_step: " << s.translation_step << ", angular_step: " << s.angular_step
       << ", window_x: " << s.window_x << ", window_y: " << s.window_y
       << ", window_yaw: " << s.window_yaw << "}" << (i < p.csm_search_stages.size() - 1 ? "," : "")
       << "\n";
  }

  os << "  ],\n"
     << "  isam_relinearizeThreshold: " << p.isam_relinearizeThreshold << ",\n"
     << "  isam_relinearizeSkip: " << p.isam_relinearizeSkip << "\n"
     << "}";

  return os;
}

}  // namespace glidar_slam::core
