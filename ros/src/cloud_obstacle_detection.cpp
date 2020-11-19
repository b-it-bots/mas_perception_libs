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
    ros::Publisher mObstacleCloudPub;
    ros::Publisher mMarkerPub;
    tf::TransformListener mTfListener;
    std::string mTransformTargetFrame;
    std::string mClusterTargetFrame;
    CloudPassThroughVoxelFilter mCloudFilter;
    EuclideanClusterParams mClusterParams;

    double mSimilarityThreshold;
    double mUniquenessThreshold;
    unsigned int mPositionCacheLimit;
    unsigned int mUniqueObstacleId;
    float mObstacleCacheTime;
    unsigned int mCurrTime;
    std::map<int, int> mLastSeenTimeCache;
    std::map<int, const PointCloud::Ptr> mObstaclesCache;
    std::map<int, std::vector<Eigen::Vector4f>> mPrevPositionsCache;

public:
    CloudObstacleDetectionNode(const ros::NodeHandle &pNodeHandle, const std::string &pCloudTopic,
            const std::string &pProcessedCloudTopic, 
            const std::string &pObstacleCloudTopic,
            const std::string &pTransformTargetFrame,
            const std::string &pClusterTargetFrame,
            const std::string &pObstaclesDetectionTopic)
    : mNodeHandle(pNodeHandle), mObstacleDetectionConfigServer(mNodeHandle), 
      mTransformTargetFrame(pTransformTargetFrame), mClusterTargetFrame(pClusterTargetFrame)
    {
        ROS_INFO("setting up dynamic reconfiguration server for obstacle detection");
        auto odCallback = boost::bind(&CloudObstacleDetectionNode::obstacleDetectionConfigCallback, this, _1, _2);
        mObstacleDetectionConfigServer.setCallback(odCallback);

        ROS_INFO("subscribing to point cloud topic and advertising processed result");
        mCloudSub = mNodeHandle.subscribe(pCloudTopic, 1, &CloudObstacleDetectionNode::cloudCallback, this);
        mFilteredCloudPub = mNodeHandle.advertise<sensor_msgs::PointCloud2>(pProcessedCloudTopic, 1);
        mObstacleCloudPub = mNodeHandle.advertise<sensor_msgs::PointCloud2>(pObstacleCloudTopic, 1);
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

        // Cloud Clustering params
        mClusterParams.mClusterTolerance = static_cast<float>(pConfig.cluster_tolerance);
        mClusterParams.mMinClusterSize = static_cast<unsigned int>(pConfig.min_cluster_size);
        mClusterParams.mMaxClusterSize = static_cast<unsigned int>(pConfig.max_cluster_size);

        // Obstacle cache params
        mObstacleCacheTime = static_cast<unsigned int>(pConfig.obstacle_cache_time);
        mSimilarityThreshold = static_cast<float>(pConfig.similarity_threshold);
        mUniquenessThreshold = static_cast<float>(pConfig.uniqueness_threshold);
        mPositionCacheLimit = static_cast<float>(pConfig.position_history_cache_size);
    }

    void
    cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& pCloudMsgPtr)
    {
        // do not process cloud when there's no subscriber
        if (mFilteredCloudPub.getNumSubscribers() == 0 && 
            mMarkerPub.getNumSubscribers() == 0 &&
            mObstacleCloudPub.getNumSubscribers() == 0)
            return;

        // transform the cloud to a desired frame
        auto transformedCloudPtr = boost::make_shared<sensor_msgs::PointCloud2>();
        if (!pcl_ros::transformPointCloud(mTransformTargetFrame, *pCloudMsgPtr, *transformedCloudPtr, mTfListener))
        {
            ROS_WARN("failed to transform cloud to frame '%s' from frame '%s'",
                     mTransformTargetFrame.c_str(), pCloudMsgPtr->header.frame_id.c_str());
            return;
        }

        // filter the cloud
        PointCloud::Ptr pclCloudPtr = boost::make_shared<PointCloud>();
        pcl::fromROSMsg(*transformedCloudPtr, *pclCloudPtr);
        PointCloud::Ptr filteredCloudPtr = mCloudFilter.filterCloud(pclCloudPtr);

        if (filteredCloudPtr->size() < mClusterParams.mMinClusterSize)
        {
            // Stop processing the point cloud if the filtered cloud does 
            // not have enough points to create even one cluster
            return;
        }

        mCurrTime = pCloudMsgPtr->header.stamp.sec;

        if (mFilteredCloudPub.getNumSubscribers() > 0)
        {
            // publish the filtered cloud for debugging
            publishCloud(filteredCloudPtr, mFilteredCloudPub);
        }

        // Euclidean clustering
        pcl::search::Search<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        tree->setInputCloud(filteredCloudPtr);
        std::vector<pcl::PointIndices> cluster_indices;
        std::vector<PointCloud::Ptr> clusterClouds;
        pcl::extractEuclideanClusters(*filteredCloudPtr,
                                      tree,
                                      mClusterParams.mClusterTolerance,
                                      cluster_indices,
                                      mClusterParams.mMinClusterSize,
                                      mClusterParams.mMaxClusterSize);
        getClusterClouds(clusterClouds, filteredCloudPtr, cluster_indices);

        processNewClusters(clusterClouds);
        removeStaleObstacles();

        publishObstacleCloud();
        publishClusterMarkers();
    }

    void
    publishClusterMarkers()
    {
        if (mMarkerPub.getNumSubscribers() <= 0)
            return;

        visualization_msgs::MarkerArray markerArray;
        for (const auto& c : mObstaclesCache)
        {
            Eigen::Vector4f min, max;
            pcl::getMinMax3D(*(c.second), min, max);
            markerArray.markers.push_back(getMarker(min, max, c.first));
            markerArray.markers.push_back(getTextLabel(min, max, c.first));
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
        marker.lifetime = ros::Duration(1.0);
        marker.header.frame_id = mTransformTargetFrame;
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

    visualization_msgs::Marker 
    getTextLabel(const Eigen::Vector4f& min, const Eigen::Vector4f& max, int id)
    {
        Eigen::Vector4f center = min + (max - min)/2.0;
        visualization_msgs::Marker marker;
        marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        marker.text = std::to_string(id);
        marker.lifetime = ros::Duration(1.0);
        marker.header.frame_id = mTransformTargetFrame;
        marker.pose.position.x = center[0];
        marker.pose.position.y = center[1];
        marker.pose.position.z = center[2];
        marker.scale.z = 0.05;
        marker.id = id + 1000;
        marker.color = std_msgs::ColorRGBA(Color(Color::SCARLET));
        return marker;
    }

    void getClusterClouds(std::vector<PointCloud::Ptr>& clusterClouds,
                          PointCloud::ConstPtr filteredCloud, 
                          const std::vector<pcl::PointIndices>& cluster_indices)
    {
        for (const auto& idx: cluster_indices)
        {
            PointCloud::Ptr clusterCloud(new PointCloud);
            for (const auto &index : idx.indices)
                clusterCloud->push_back ((*filteredCloud)[index]);

            clusterCloud->header = filteredCloud->header;
            clusterCloud->width = clusterCloud->size();
            clusterCloud->height = 1;
            clusterCloud->is_dense = true;

            PointCloud::Ptr transformedCloudPtr = boost::make_shared<PointCloud>();
            pcl_ros::transformPointCloud(mClusterTargetFrame, *clusterCloud, *transformedCloudPtr, mTfListener);

            clusterClouds.push_back(transformedCloudPtr);
        }
    }

    // Cloud comparison sample from https://stackoverflow.com/a/55930847
    float nearestDistance(const pcl::search::KdTree<PointT>& tree, const PointT& pt)
    {
        const int k = 1;
        std::vector<int> indices (k);
        std::vector<float> sqr_distances (k);

        tree.nearestKSearch(pt, k, indices, sqr_distances);

        return sqr_distances[0];
    }
    // compare cloudB to cloudA
    // use threshold for identifying outliers and not considering those for the similarity
    // a good value for threshold is 5 * <cloud_resolution>, e.g. 10cm for a cloud with 2cm resolution
    float getCloudSimilarity(const PointCloud& cloudA, const PointCloud& cloudB, float threshold)
    {
        // compare B to A
        int num_outlier = 0;
        pcl::search::KdTree<PointT> tree;
        tree.setInputCloud(cloudA.makeShared());
        auto sum = std::accumulate(cloudB.begin(), cloudB.end(), 0.0f, [&](float current_sum, const PointT& pt) {
            const auto dist = nearestDistance(tree, pt);

            if(dist < threshold)
            {
                return current_sum + dist;
            }
            else
            {
                num_outlier++;
                return current_sum;
            }
        });

        return sum / (cloudB.size() - num_outlier);
    }

    bool 
    isNewObstacle(const PointCloud& cloud, int& knownObstacleId)
    {
        for (const auto& o : mObstaclesCache)
        {
            if (getCloudSimilarity(cloud, *(o.second), mSimilarityThreshold) < mUniquenessThreshold)
            {
                knownObstacleId = o.first;
                return false;
            }
        }
        knownObstacleId = -1;
        return true;
    }

    void
    processNewClusters(const std::vector<PointCloud::Ptr>& clusterClouds)
    {
        for (const auto& cloud : clusterClouds)
        {
            Eigen::Vector4f centroid;
            if (pcl::compute3DCentroid(*cloud, centroid) <= 0)
            {
                ROS_WARN("Could not compute centroid of point cloud. Skipping processing of potential obstacle cluster!");
                return;
            }

            int obstacleID = -1;
            if (isNewObstacle(*cloud, obstacleID))
            {
                obstacleID = mUniqueObstacleId++;
                ROS_INFO("Adding new obstacle: %d", obstacleID);
                mObstaclesCache.insert(std::pair<int, const PointCloud::Ptr>(obstacleID, cloud));
                std::vector<Eigen::Vector4f> prevPositions;
                prevPositions.push_back(centroid);
                mPrevPositionsCache.insert(std::pair<int, std::vector<Eigen::Vector4f>>(obstacleID, prevPositions));
                mLastSeenTimeCache.insert(std::pair<int, int>(obstacleID, mCurrTime));
            }
            else
            {
                // update last known position
                std::vector<Eigen::Vector4f>& prevPositions = mPrevPositionsCache[obstacleID];
                if (prevPositions.size() > mPositionCacheLimit)
                {
                    prevPositions.erase(prevPositions.begin());
                }
                prevPositions.push_back(centroid);

                mLastSeenTimeCache[obstacleID] = mCurrTime;
            }
        }
    }

    void removeStaleObstacles()
    {
        for (auto it = mLastSeenTimeCache.cbegin(); it != mLastSeenTimeCache.cend(); )
        {
            int id = it->first;
            if (mCurrTime - it->second > mObstacleCacheTime)
            {
                ROS_INFO("Removing Stale obstacle: %d", id);
                mLastSeenTimeCache.erase(it++);
                mPrevPositionsCache.erase(id);
                mObstaclesCache.erase(id);
            }
            else
            {
                ++it;
            }
        }
    }

    void
    publishObstacleCloud()
    {
        if (mObstacleCloudPub.getNumSubscribers() <= 0)
            return;

        // Merge all the clusters into once cloud
        PointCloud::Ptr mergedCloud(new PointCloud);
        bool headerInitialized = false;
        for (const auto& c : mObstaclesCache)
        {   if (!headerInitialized)
            {
                mergedCloud->header = c.second->header;
                headerInitialized = true;
            }
            *mergedCloud += *(c.second);
        }
        mergedCloud->width = mergedCloud->size();
        mergedCloud->height = 1;
        mergedCloud->is_dense = true;

        // publish the cloud
        publishCloud(mergedCloud, mObstacleCloudPub);
    }

    void
    publishCloud(PointCloud::ConstPtr cloudPtr, const ros::Publisher& publisher)
    {
        // publish the cloud
        sensor_msgs::PointCloud2::Ptr cloudMsgPtr = boost::make_shared<sensor_msgs::PointCloud2>();
        pcl::toROSMsg(*cloudPtr, *cloudMsgPtr);
        publisher.publish(*cloudMsgPtr);
    }
};

}   // namespace mas_perception_libs

int main(int pArgc, char** pArgv)
{
    ros::init(pArgc, pArgv, "cloud_obstacle_detection");
    ros::NodeHandle nh("~");

    // load launch parameters
    std::string cloudTopic, processedCloudTopic, obstacleCloudTopic, 
                obstacleDetectionsTopic, transformTargetFrame, clusterTargetFrame;
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
    if (!nh.getParam("obstacle_cloud_topic", obstacleCloudTopic) || obstacleCloudTopic.empty())
    {
        ROS_ERROR("No 'obstacle_cloud_topic' specified as parameter");
        return EXIT_FAILURE;
    }
    if (!nh.getParam("obstacles_detection_topic", obstacleDetectionsTopic) || obstacleDetectionsTopic.empty())
    {
        ROS_ERROR("No 'obstacles_detection_topic' specified as parameter");
        return EXIT_FAILURE;
    }
    if (!nh.getParam("transform_target_frame", transformTargetFrame) || transformTargetFrame.empty())
    {
        ROS_ERROR("No 'transform_target_frame' specified as parameter");
        return EXIT_FAILURE;
    }
    if (!nh.getParam("cluster_target_frame", clusterTargetFrame) || clusterTargetFrame.empty())
    {
        ROS_ERROR("No 'cluster_target_frame' specified as parameter");
        return EXIT_FAILURE;
    }

    // run cloud filtering and obstacle detection
    mas_perception_libs::CloudObstacleDetectionNode obstacleDetection(nh, cloudTopic, processedCloudTopic, obstacleCloudTopic,
                                                                         transformTargetFrame, clusterTargetFrame, obstacleDetectionsTopic);

    while (ros::ok())
        ros::spin();

    return 0;
}
