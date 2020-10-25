/*!
 * @copyright 2018 Bonn-Rhein-Sieg University
 *
 * @author Sushant Chavan
 *
 * @brief script to detect obstacles from a point cloud
 */

#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <dynamic_reconfigure/server.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>
#include <mas_perception_libs/color.h>
#include <mas_perception_libs/ObstacleDetectionConfig.h>

namespace mas_perception_libs
{

/*!
 * @brief struct containing parameters necessary for filtering point clouds
 */
struct CloudPassThroughVoxelFilterParams
{
    /* PassThrough filter parameters
     * limit the cloud to be filtered points outside of these x, y, z ranges will be discarded */
    float mPassThroughLimitMinX = 0.0f;
    float mPassThroughLimitMaxX = 0.0f;
    float mPassThroughLimitMinY = 0.0f;
    float mPassThroughLimitMaxY = 0.0f;
    float mPassThroughLimitMinZ = 0.0f;
    float mPassThroughLimitMaxZ = 0.0f;
    /* VoxelGrid filter parameters for down-sampling the cloud, also limit the cloud along the z axis */
    float mVoxelLimitMinZ = 0.0f;
    float mVoxelLimitMaxZ = 0.0f;
    float mVoxelLeafSize = 0.0f;
};

/*!
 * @brief struct containing parameters necessary for clustering point clouds
 */
struct EuclideanClusterParams
{
    float mClusterTolerance = 0.01f;
    unsigned int mMinClusterSize = 1;
    unsigned int mMaxClusterSize = std::numeric_limits<unsigned int>::max();
};

/*!
 * @brief class containing definition for filtering point clouds
 */
class CloudPassThroughVoxelFilter
{
public:
    CloudPassThroughVoxelFilter() = default;

    /*! @brief set parameters relevant to filtering cloud */
    virtual void
    setParams(const CloudPassThroughVoxelFilterParams& pParams)
    {
        /* pass-through params */
        mPassThroughFilterX.setFilterFieldName("x");
        mPassThroughFilterX.setFilterLimits(pParams.mPassThroughLimitMinX, pParams.mPassThroughLimitMaxX);
        mPassThroughFilterY.setFilterFieldName("y");
        mPassThroughFilterY.setFilterLimits(pParams.mPassThroughLimitMinY, pParams.mPassThroughLimitMaxY);
        mPassThroughFilterZ.setFilterFieldName("z");
        mPassThroughFilterZ.setFilterLimits(pParams.mPassThroughLimitMinZ, pParams.mPassThroughLimitMaxZ);

        /* filter z-axis using voxel filter instead of making another member */
        mVoxelGridFilter.setFilterFieldName("z");
        mVoxelGridFilter.setFilterLimits(pParams.mVoxelLimitMinZ, pParams.mVoxelLimitMaxZ);

        /* voxel-grid params */
        mVoxelGridFilter.setLeafSize(pParams.mVoxelLeafSize, pParams.mVoxelLeafSize, pParams.mVoxelLeafSize);
    }

    /*!
    * @brief filter point cloud using passthrough and voxel filters
    */
    PointCloud::Ptr
    filterCloud(const PointCloud::ConstPtr &pCloudPtr)
    {
        PointCloud::Ptr filteredCloudPtr = boost::make_shared<PointCloud>();

        mPassThroughFilterX.setInputCloud(pCloudPtr);
        mPassThroughFilterX.filter(*filteredCloudPtr);

        mPassThroughFilterY.setInputCloud(filteredCloudPtr);
        mPassThroughFilterY.filter(*filteredCloudPtr);

        mPassThroughFilterZ.setInputCloud(filteredCloudPtr);
        mPassThroughFilterZ.filter(*filteredCloudPtr);

        mVoxelGridFilter.setInputCloud(filteredCloudPtr);
        mVoxelGridFilter.filter(*filteredCloudPtr);

        return filteredCloudPtr;
    }

private:
    pcl::PassThrough<PointT> mPassThroughFilterX;
    pcl::PassThrough<PointT> mPassThroughFilterY;
    pcl::PassThrough<PointT> mPassThroughFilterZ;
    pcl::VoxelGrid<PointT> mVoxelGridFilter;
};

class CloudObstacleDetectionNode
{
private:
    ros::NodeHandle mNodeHandle;
    dynamic_reconfigure::Server<ObstacleDetectionConfig> mObstacleDetectionConfigServer;
    ros::Subscriber mCloudSub;
    ros::Publisher mFilteredCloudPub;
    ros::Publisher mMarkerPub;
    tf::TransformListener mTfListener;
    std::string mTargetFrame;
    CloudPassThroughVoxelFilter mCloudFilter;
    EuclideanClusterParams mClusterParams;

public:
    CloudObstacleDetectionNode(const ros::NodeHandle &pNodeHandle, const std::string &pCloudTopic,
            const std::string &pProcessedCloudTopic, 
            const std::string &pTargetFrame, 
            const std::string &pObstaclesDetectionTopic)
    : mNodeHandle(pNodeHandle), mObstacleDetectionConfigServer(mNodeHandle), mTargetFrame(pTargetFrame)
    {
        ROS_INFO("setting up dynamic reconfiguration server for obstacle detection");
        auto odCallback = boost::bind(&CloudObstacleDetectionNode::obstacleDetectionConfigCallback, this, _1, _2);
        mObstacleDetectionConfigServer.setCallback(odCallback);

        ROS_INFO("subscribing to point cloud topic and advertising processed result");
        mCloudSub = mNodeHandle.subscribe(pCloudTopic, 1, &CloudObstacleDetectionNode::cloudCallback, this);
        mFilteredCloudPub = mNodeHandle.advertise<sensor_msgs::PointCloud2>(pProcessedCloudTopic, 1);
        mMarkerPub = mNodeHandle.advertise<visualization_msgs::MarkerArray>(pObstaclesDetectionTopic, 1);
    }

private:
    void
    obstacleDetectionConfigCallback(const ObstacleDetectionConfig &pConfig, uint32_t pLevel)
    {
        // Cloud Filter params
        CloudPassThroughVoxelFilterParams cloudFilterParams;
        cloudFilterParams.mPassThroughLimitMinX = static_cast<float>(pConfig.passthrough_limit_min_x);
        cloudFilterParams.mPassThroughLimitMaxX = static_cast<float>(pConfig.passthrough_limit_max_x);
        cloudFilterParams.mPassThroughLimitMinY = static_cast<float>(pConfig.passthrough_limit_min_y);
        cloudFilterParams.mPassThroughLimitMaxY = static_cast<float>(pConfig.passthrough_limit_max_y);
        cloudFilterParams.mPassThroughLimitMinZ = static_cast<float>(pConfig.passthrough_limit_min_z);
        cloudFilterParams.mPassThroughLimitMaxZ = static_cast<float>(pConfig.passthrough_limit_max_z);
        cloudFilterParams.mVoxelLimitMinZ = static_cast<float>(pConfig.voxel_limit_min_z);
        cloudFilterParams.mVoxelLimitMaxZ = static_cast<float>(pConfig.voxel_limit_max_z);
        cloudFilterParams.mVoxelLeafSize = static_cast<float>(pConfig.voxel_leaf_size);
        mCloudFilter.setParams(cloudFilterParams);

        // CLoud CLustering params
        mClusterParams.mClusterTolerance = static_cast<float>(pConfig.cluster_tolerance);
        mClusterParams.mMinClusterSize = static_cast<unsigned int>(pConfig.min_cluster_size);
        mClusterParams.mMaxClusterSize = static_cast<unsigned int>(pConfig.max_cluster_size);
    }

    void
    cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& pCloudMsgPtr)
    {
        // do not process cloud when there's no subscriber
        if (mFilteredCloudPub.getNumSubscribers() == 0 && mMarkerPub.getNumSubscribers() == 0)
            return;

        // transform the cloud to a desired frame
        auto transformedCloudPtr = boost::make_shared<sensor_msgs::PointCloud2>();
        if (!pcl_ros::transformPointCloud(mTargetFrame, *pCloudMsgPtr, *transformedCloudPtr, mTfListener))
        {
            ROS_WARN("failed to transform cloud to frame '%s' from frame '%s'",
                     mTargetFrame.c_str(), pCloudMsgPtr->header.frame_id.c_str());
            return;
        }

        // filter the cloud
        PointCloud::Ptr pclCloudPtr = boost::make_shared<PointCloud>();
        pcl::fromROSMsg(*transformedCloudPtr, *pclCloudPtr);
        PointCloud::Ptr filteredCloudPtr = mCloudFilter.filterCloud(pclCloudPtr);

        // publish the filtered cloud for debugging
        sensor_msgs::PointCloud2::Ptr filteredMsgPtr = boost::make_shared<sensor_msgs::PointCloud2>();
        pcl::toROSMsg(*filteredCloudPtr, *filteredMsgPtr);
        mFilteredCloudPub.publish(*filteredMsgPtr);

        // Euclidean clustering
        pcl::search::Search<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        tree->setInputCloud(filteredCloudPtr);
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::extractEuclideanClusters(*filteredCloudPtr,
                                      tree,
                                      mClusterParams.mClusterTolerance,
                                      cluster_indices,
                                      mClusterParams.mMinClusterSize,
                                      mClusterParams.mMaxClusterSize);

        publishClusterMarkers(filteredCloudPtr, cluster_indices);
    }

    void 
    publishClusterMarkers(PointCloud::ConstPtr filteredCloud, 
                          const std::vector<pcl::PointIndices>& cluster_indices)
    {
        visualization_msgs::MarkerArray markerArray;
        int i = 0;
        for (const auto& indices: cluster_indices)
        {
            Eigen::Vector4f min, max;
            pcl::getMinMax3D(*filteredCloud, indices, min, max);
            markerArray.markers.push_back(getMarker(min, max, i++));
        }
        // Publish the marker array
        mMarkerPub.publish(markerArray);
    }

    geometry_msgs::Point 
    getGeomPoint(float x, float y, float z)
    {
        geometry_msgs::Point p;
        p.x = x;
        p.y = y;
        p.z = z;
        return p;
    }

    visualization_msgs::Marker 
    getMarker(const Eigen::Vector4f& min, const Eigen::Vector4f& max, int id)
    {
        visualization_msgs::Marker marker;
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.lifetime = ros::Duration(2.0);
        marker.header.frame_id = mTargetFrame;
        marker.scale.x = 0.005;
        marker.scale.y = 0.005;
        marker.color.a = 2.0;
        marker.ns = "";
        marker.id = id;
        marker.color = std_msgs::ColorRGBA(Color(Color::SCARLET));
        marker.points.push_back(getGeomPoint(min[0], min[1], min[2]));
        marker.points.push_back(getGeomPoint(min[0], max[1], min[2]));
        marker.points.push_back(getGeomPoint(min[0], max[1], min[2]));
        marker.points.push_back(getGeomPoint(max[0], max[1], min[2]));
        marker.points.push_back(getGeomPoint(max[0], max[1], min[2]));
        marker.points.push_back(getGeomPoint(max[0], min[1], min[2]));
        marker.points.push_back(getGeomPoint(max[0], min[1], min[2]));
        marker.points.push_back(getGeomPoint(min[0], min[1], min[2]));
        return marker;
    }
};

}   // namespace mas_perception_libs

int main(int pArgc, char** pArgv)
{
    ros::init(pArgc, pArgv, "cloud_obstacle_detection");
    ros::NodeHandle nh("~");

    // load launch parameters
    std::string cloudTopic, processedCloudTopic, obstacleDetectionsTopic, targetFrame;
    if (!nh.getParam("cloud_topic", cloudTopic) || cloudTopic.empty())
    {
        ROS_ERROR("No 'cloud_topic' specified as parameter");
        return EXIT_FAILURE;
    }
    if (!nh.getParam("processed_cloud_topic", processedCloudTopic) || processedCloudTopic.empty())
    {
        ROS_ERROR("No 'processed_cloud_topic' specified as parameter");
        return EXIT_FAILURE;
    }
    if (!nh.getParam("obstacles_detection_topic", obstacleDetectionsTopic) || obstacleDetectionsTopic.empty())
    {
        ROS_ERROR("No 'obstacles_detection_topic' specified as parameter");
        return EXIT_FAILURE;
    }
    if (!nh.getParam("target_frame", targetFrame) || targetFrame.empty())
    {
        ROS_ERROR("No 'target_frame' specified as parameter");
        return EXIT_FAILURE;
    }

    // run cloud filtering and obstacle detection
    mas_perception_libs::CloudObstacleDetectionNode obstacleDetection(nh, cloudTopic, processedCloudTopic,
                                                                         targetFrame, obstacleDetectionsTopic);

    while (ros::ok())
        ros::spin();

    return 0;
}
