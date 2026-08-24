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
  std::string csm_debug_low_topic{"/glidar_slam/debug/csm_likelihood_low"};
  std::string csm_debug_high_topic{"/glidar_slam/debug/csm_likelihood_high"};
  std::string color_image_topic{"/camera/color/image_raw"};
  std::string aligned_depth_image_topic{"/camera/aligned_depth/image_raw"};
  std::string color_camera_info_topic{"/camera/color/camera_info"};

  std::vector<double> odom_covariance_diagonal{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0};

  // General Parameters
  double minimum_travel_distance{0.5};
  double minimum_travel_heading{0.5};

  // Occupancy Grid Parameters
  double occ_map_resolution{0.05};
  int occ_map_padding{2};

  // Submap Parameters
  int submap_window_size{5};

  // Ground Observation Parameters
  bool ground_debug_enable{false};
  std::string ground_debug_cloud_topic{"glidar_slam/debug/ground_inliers"};
  std::string ground_debug_marker_topic{"glidar_slam/debug/ground_plane"};
  std::string ground_debug_image_topic{"glidar_slam/debug/ground_overlay"};
  double ground_debug_plane_size{4.0};
  bool ground_optimization_enable{true};
  double ground_observation_max_age_sec{0.1};
  int ground_minimum_inlier_count{100};
  double ground_normal_sigma{0.05};
  double ground_distance_sigma{0.03};
  double ground_fallback_variance{1.0};
  std::vector<float> ground_roi_ratios{0.0F, 0.0F, 0.2F, 0.4F};
  int ground_extraction_pixel_stride{2};
  double ground_extraction_voxel_size{0.05};
  double ground_extraction_max_distance{5.0};
  double ground_extraction_distance_threshold{0.05};
  bool ground_mapping_enable_texture_mapping{true};
  bool ground_mapping_enable_ground_marking_mapping{false};
  std::string ground_marking_map_topic{"ground_markings_map"};
  std::string ground_texture_map_topic{"ground_texture_map"};
  std::string ground_texture_coverage_topic{"ground_texture_coverage"};
  double ground_map_resolution{0.02};
  int ground_map_padding{2};
  int ground_marking_white_threshold{200};
  bool ground_matching_enable{false};
  int ground_matching_minimum_marking_count{20};
  double ground_matching_minimum_score{0.5};
  double ground_matching_min_forward_distance{0.0};
  double ground_matching_max_forward_distance{2.0};
  double ground_matching_csm_smear_deviation{0.03};
  bool ground_matching_debug_enable{false};
  std::string ground_matching_debug_topic{"glidar_slam/debug/ground_matching"};

  // Correlative Scan Matcher Parameters
  double csm_smear_deviation{0.1};
  bool csm_use_distance_transform{false};
  bool csm_use_tbb{false};
  bool csm_use_penalty{true};
  double csm_distance_penalty_std_dev{0.5};
  double csm_angle_penalty_std_dev{1.0};
  std::vector<CsmSearchStage> csm_search_stages;
  bool csm_debug_enable{false};

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
  bool debug_timings{false};

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
     << "  csm_debug_low_topic: " << p.csm_debug_low_topic << ",\n"
     << "  csm_debug_high_topic: " << p.csm_debug_high_topic << ",\n"
     << "  color_image_topic: " << p.color_image_topic << ",\n"
     << "  aligned_depth_image_topic: " << p.aligned_depth_image_topic << ",\n"
     << "  color_camera_info_topic: " << p.color_camera_info_topic << ",\n"
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
     << "  ground_debug_enable: " << (p.ground_debug_enable ? "true" : "false") << ",\n"
     << "  ground_debug_cloud_topic: " << p.ground_debug_cloud_topic << ",\n"
     << "  ground_debug_marker_topic: " << p.ground_debug_marker_topic << ",\n"
     << "  ground_debug_image_topic: " << p.ground_debug_image_topic << ",\n"
     << "  ground_debug_plane_size: " << p.ground_debug_plane_size << ",\n"
     << "  ground_optimization_enable: " << (p.ground_optimization_enable ? "true" : "false")
     << ",\n"
     << "  ground_observation_max_age_sec: " << p.ground_observation_max_age_sec << ",\n"
     << "  ground_minimum_inlier_count: " << p.ground_minimum_inlier_count << ",\n"
     << "  ground_normal_sigma: " << p.ground_normal_sigma << ",\n"
     << "  ground_distance_sigma: " << p.ground_distance_sigma << ",\n"
     << "  ground_fallback_variance: " << p.ground_fallback_variance << ",\n"
     << "  ground_roi_ratios: [";

  for (std::size_t i = 0; i < p.ground_roi_ratios.size(); ++i) {
    os << p.ground_roi_ratios[i] << (i < p.ground_roi_ratios.size() - 1 ? ", " : "");
  }

  os << "],\n"
     << "  ground_extraction_pixel_stride: " << p.ground_extraction_pixel_stride << ",\n"
     << "  ground_extraction_voxel_size: " << p.ground_extraction_voxel_size
     << ",\n"
     //  << "  ground_extraction_min_depth: " << p.ground_extraction_min_depth << ",\n"
     << "  ground_extraction_max_distance: " << p.ground_extraction_max_distance << ",\n"
     << "  ground_extraction_distance_threshold: " << p.ground_extraction_distance_threshold
     << ",\n"
     << "  ground_mapping_enable_texture_mapping: "
     << (p.ground_mapping_enable_texture_mapping ? "true" : "false") << ",\n"
     << "  ground_mapping_enable_ground_marking_mapping: "
     << (p.ground_mapping_enable_ground_marking_mapping ? "true" : "false") << ",\n"
     << "  ground_marking_map_topic: " << p.ground_marking_map_topic << ",\n"
     << "  ground_texture_map_topic: " << p.ground_texture_map_topic << ",\n"
     << "  ground_texture_coverage_topic: " << p.ground_texture_coverage_topic << ",\n"
     << "  ground_map_resolution: " << p.ground_map_resolution << ",\n"
     << "  ground_map_padding: " << p.ground_map_padding << ",\n"
     << "  ground_marking_white_threshold: " << p.ground_marking_white_threshold << ",\n"
     << "  ground_matching_enable: " << (p.ground_matching_enable ? "true" : "false") << ",\n"
     << "  ground_matching_minimum_marking_count: " << p.ground_matching_minimum_marking_count
     << ",\n"
     << "  ground_matching_minimum_score: " << p.ground_matching_minimum_score << ",\n"
     << "  ground_matching_debug_enable: " << (p.ground_matching_debug_enable ? "true" : "false")
     << ",\n"
     << "  ground_matching_debug_topic: " << p.ground_matching_debug_topic << ",\n"
     << "  csm_debug_enable: " << (p.csm_debug_enable ? "true" : "false") << ",\n"
     << "  csm_smear_deviation: " << p.csm_smear_deviation << ",\n"
     << "  csm_use_distance_transform: " << (p.csm_use_distance_transform ? "true" : "false")
     << ",\n"
     << "  csm_use_tbb: " << (p.csm_use_tbb ? "true" : "false") << ",\n"
     << "  csm_use_penalty: " << (p.csm_use_penalty ? "true" : "false") << ",\n"
     << "  csm_distance_penalty_std_dev: " << p.csm_distance_penalty_std_dev << ",\n"
     << "  csm_angle_penalty_std_dev: " << p.csm_angle_penalty_std_dev << ",\n"
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
     << "  debug_timings: " << (p.debug_timings ? "true" : "false") << ",\n"
     << "  isam_relinearizeThreshold: " << p.isam_relinearizeThreshold << ",\n"
     << "  isam_relinearizeSkip: " << p.isam_relinearizeSkip << "\n"
     << "}";

  return os;
}

}  // namespace glidar_slam::core
