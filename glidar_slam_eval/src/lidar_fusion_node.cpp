#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "message_filters/subscriber.hpp"
#include "message_filters/sync_policies/approximate_time.hpp"
#include "message_filters/synchronizer.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "tf2/LinearMath/Transform.hpp"
#include "tf2/exceptions.hpp"
#include "tf2/time.hpp"
#include "tf2_ros/buffer.hpp"
#include "tf2_ros/transform_listener.hpp"

namespace glidar_slam_eval {

class LidarFusionNode : public rclcpp::Node
{
public:
  LidarFusionNode() : Node("lidar_fusion")
  {
    front_scan_topic_ =
      this->declare_parameter<std::string>("front_scan_topic", "/laser_scan_front/scan");
    rear_scan_topic_ =
      this->declare_parameter<std::string>("rear_scan_topic", "/laser_scan_rear/scan");
    output_topic_ = this->declare_parameter<std::string>("output_topic", "/scan");
    sync_queue_size_ = this->declare_parameter<int>("sync_queue_size", 50);
    sync_max_interval_seconds_ = this->declare_parameter<double>("sync_max_interval", 0.1);
    tf_lookup_timeout_seconds_ = this->declare_parameter<double>("tf_lookup_timeout", 0.1);

    if (
      sync_queue_size_ <= 0 || !std::isfinite(sync_max_interval_seconds_) ||
      sync_max_interval_seconds_ < 0.0 || !std::isfinite(tf_lookup_timeout_seconds_) ||
      tf_lookup_timeout_seconds_ < 0.0) {
      throw std::runtime_error("Invalid LiDAR synchronization or TF parameters");
    }

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    fused_scan_publisher_ = this->create_publisher<sensor_msgs::msg::LaserScan>(output_topic_, 10);

    front_scan_subscriber_.subscribe(this, front_scan_topic_, 10);
    rear_scan_subscriber_.subscribe(this, rear_scan_topic_, 10);

    synchronizer_ = std::make_shared<message_filters::Synchronizer<ScanSyncPolicy>>(
      ScanSyncPolicy(sync_queue_size_), front_scan_subscriber_, rear_scan_subscriber_);
    synchronizer_->registerCallback(
      std::bind(
        &LidarFusionNode::scanCallback, this, std::placeholders::_1, std::placeholders::_2));
    synchronizer_->setMaxIntervalDuration(
      rclcpp::Duration::from_seconds(sync_max_interval_seconds_));

    RCLCPP_INFO(
      this->get_logger(), "Fusing '%s' and '%s' into '%s'", front_scan_topic_.c_str(),
      rear_scan_topic_.c_str(), output_topic_.c_str());
  }

private:
  using Scan = sensor_msgs::msg::LaserScan;
  using ScanSyncPolicy = message_filters::sync_policies::ApproximateTime<Scan, Scan>;

  struct PointWithIntensity
  {
    tf2::Vector3 point;
    float intensity;
  };

  static tf2::Transform transformFromMessage(
    const geometry_msgs::msg::TransformStamped & transform_message)
  {
    const auto & translation = transform_message.transform.translation;
    const auto & rotation = transform_message.transform.rotation;
    return tf2::Transform(
      tf2::Quaternion(rotation.x, rotation.y, rotation.z, rotation.w),
      tf2::Vector3(translation.x, translation.y, translation.z));
  }

  static void appendScanPoints(
    const Scan & scan, const tf2::Transform & scan_to_front,
    std::vector<PointWithIntensity> & points)
  {
    double angle = scan.angle_min;
    for (std::size_t index = 0; index < scan.ranges.size(); ++index) {
      const float range = scan.ranges[index];
      if (std::isfinite(range) && range >= scan.range_min && range <= scan.range_max) {
        const float intensity = index < scan.intensities.size() ? scan.intensities[index] : 0.0F;
        points.push_back(
          {scan_to_front * tf2::Vector3(range * std::cos(angle), range * std::sin(angle), 0.0),
           intensity});
      }
      angle += scan.angle_increment;
    }
  }

  sensor_msgs::msg::LaserScan::UniquePtr fuseScans(
    const Scan & front_scan, const Scan & rear_scan, const tf2::Transform & rear_to_front) const
  {
    const double angular_increment =
      std::min(std::abs(front_scan.angle_increment), std::abs(rear_scan.angle_increment));
    if (!std::isfinite(angular_increment) || angular_increment <= 0.0) {
      return nullptr;
    }

    std::vector<PointWithIntensity> points;
    points.reserve(front_scan.ranges.size() + rear_scan.ranges.size());
    appendScanPoints(front_scan, tf2::Transform::getIdentity(), points);
    appendScanPoints(rear_scan, rear_to_front, points);
    if (points.empty()) {
      return nullptr;
    }

    double angle_min = std::numeric_limits<double>::infinity();
    double angle_max = -std::numeric_limits<double>::infinity();
    double max_range = 0.0;
    for (const auto & point_with_intensity : points) {
      const auto & point = point_with_intensity.point;
      const double angle = std::atan2(point.y(), point.x());
      angle_min = std::min(angle_min, angle);
      angle_max = std::max(angle_max, angle);
      max_range = std::max(max_range, std::hypot(point.x(), point.y()));
    }

    const auto bin_count =
      static_cast<std::size_t>(std::ceil((angle_max - angle_min) / angular_increment) + 1.0);
    if (bin_count == 0 || bin_count > 1000000U) {
      return nullptr;
    }

    auto fused_scan = std::make_unique<Scan>();
    fused_scan->header = front_scan.header;
    fused_scan->angle_min = static_cast<float>(angle_min);
    fused_scan->angle_increment = static_cast<float>(angular_increment);
    fused_scan->angle_max = static_cast<float>(angle_min + angular_increment * (bin_count - 1));
    fused_scan->time_increment = front_scan.time_increment;
    fused_scan->scan_time = front_scan.scan_time;
    fused_scan->range_min = std::min(front_scan.range_min, rear_scan.range_min);
    fused_scan->range_max = static_cast<float>(std::max(
      {static_cast<double>(front_scan.range_max), static_cast<double>(rear_scan.range_max),
       max_range}));
    fused_scan->ranges.assign(bin_count, std::numeric_limits<float>::infinity());
    fused_scan->intensities.assign(bin_count, 0.0F);

    for (const auto & point_with_intensity : points) {
      const auto & point = point_with_intensity.point;
      const double point_angle = std::atan2(point.y(), point.x());
      const double point_range = std::hypot(point.x(), point.y());
      const auto bin =
        static_cast<std::size_t>(std::llround((point_angle - angle_min) / angular_increment));
      if (bin < fused_scan->ranges.size() && point_range < fused_scan->ranges[bin]) {
        fused_scan->ranges[bin] = static_cast<float>(point_range);
        fused_scan->intensities[bin] = point_with_intensity.intensity;
      }
    }

    return fused_scan;
  }

  void scanCallback(const Scan::ConstSharedPtr & front_scan, const Scan::ConstSharedPtr & rear_scan)
  {
    const std::string front_frame =
      front_scan->header.frame_id.empty() ? "scan_front_link" : front_scan->header.frame_id;
    const std::string rear_frame =
      rear_scan->header.frame_id.empty() ? "scan_rear_link" : rear_scan->header.frame_id;
    try {
      const auto rear_to_front_message = tf_buffer_->lookupTransform(
        front_frame, rear_frame, tf2::TimePointZero,
        tf2::durationFromSec(tf_lookup_timeout_seconds_));
      const auto fused_scan =
        fuseScans(*front_scan, *rear_scan, transformFromMessage(rear_to_front_message));
      if (!fused_scan) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Dropping synchronized LiDAR pair with invalid scan geometry");
        return;
      }
      fused_scan_publisher_->publish(std::move(*fused_scan));
    } catch (const tf2::TransformException & exception) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Dropping synchronized LiDAR pair: TF %s <- %s unavailable: %s", front_frame.c_str(),
        rear_frame.c_str(), exception.what());
    }
  }

  std::string front_scan_topic_;
  std::string rear_scan_topic_;
  std::string output_topic_;
  int sync_queue_size_{50};
  double sync_max_interval_seconds_{0.1};
  double tf_lookup_timeout_seconds_{0.1};
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  message_filters::Subscriber<Scan> front_scan_subscriber_;
  message_filters::Subscriber<Scan> rear_scan_subscriber_;
  std::shared_ptr<message_filters::Synchronizer<ScanSyncPolicy>> synchronizer_;
  rclcpp::Publisher<Scan>::SharedPtr fused_scan_publisher_;
};

}  // namespace glidar_slam_eval

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<glidar_slam_eval::LidarFusionNode>());
  rclcpp::shutdown();
  return 0;
}
