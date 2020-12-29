# `mas_perception_libs`

Library containing shared perception functionality. Python interface are developed for several functions and definitions
in the C++ library using [Boost](https://www.boost.org/doc/libs/1_68_0/libs/python/doc/html/index.html). This is to
avoid defining commonly used logic in different locations and also to expose C++ exclusive libraries (i.e. PCL) for use
in Python. This means if a functionality is needed in both Python and C++, it should first be developed in C++ and then
ported to Python.

- [`mas_perception_libs`](#mas_perception_libs)
  - [Build issues](#build-issues)
  - [C++ Library](#c-library)
  - [Python Package](#python-package)
  - [Executables](#executables)
    - [`object_detection_action_server`](#object_detection_action_server)
    - [`plane_detection_action_server`](#plane_detection_action_server)
    - [`image_detection_test`](#image_detection_test)
    - [`image_recognition_server`](#image_recognition_server)
    - [`image_recognition_client_test`](#image_recognition_client_test)
    - [`cloud_processing_python_test`](#cloud_processing_python_test)
    - [`cloud_processing_cpp_test`](#cloud_processing_cpp_test)
  - [Launch Files](#launch-files)
    - [`plane_detection.launch`](#plane_detectionlaunch)
    - [`object_detection.launch`](#object_detectionlaunch)
    - [`image_detection_test.launch`](#image_detection_testlaunch)
    - [`image_recognition.launch`](#image_recognitionlaunch)
    - [`cloud_processing_python_test.launch`](#cloud_processing_python_testlaunch)
    - [`cloud_processing_cpp_test.launch`](#cloud_processing_cpp_testlaunch)
  - [Directory structure](#directory-structure)

## Build issues

* `numpy` installation may not create a correct symbolic link to the library's include directory in the system, which
  results in a compilation error about missing header file `numpy/arrayobject.h`. This can be fixed by creating a correct
  link as follow (this is typical for an Ubuntu machine, please make sure it's appropriate for your own setup):
  ```
  # ln -s /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy /usr/include/numpy
  ```

## [C++ Library](docs/cpp_library.md)

Contains shared perception definitions in C++.

## [Python Package](docs/python_package.md)

Contains shared perception definitions in Python.

## Executables

### [`object_detection_action_server`](ros/scripts/object_detection_action_server)
Run an instance of `ObjectDetectionActionServer` (described in [Python Documentation](docs/python_package.md)).

Parameters:
* `action_name`: name of action server
* `cloud_topic`: name of topic which supply `sensor_msgs/PointCloud2` messages
* `target_frame`: name of reference frame which the object poses will be transformed to
* `class_annotations`: YAML file which maps numeric class values to class names, used by the
[`ImageDetectorBase` class](docs/python_package.md) to configure the detection model. An example of this file is
[`class_annotation_example.yml`](models/class_annotation_example.yml).
* `kwargs_file`: YAML file which is used by the [`ImageDetectorBase` class](docs/python_package.md) to specify
additional parameters needed to configure the detection model. An example is
[`image_detector_test_kwargs.yml`](models/image_detector_test_kwargs.yml).
* `detection_module`: name of the module containing the `ImageDetectorBase` extension to import.
* `detection_class`: name of the extension of the `ImageDetectorBase` class to import.


### [`plane_detection_action_server`](ros/scripts/plane_detection_action_server)
Run an instance of `PlaneDetectionActionServer` (described in [Python Documentation](docs/python_package.md)).

Parameters:
* `action_name`: name of action server
* `cloud_topic`: name of topic which supply `sensor_msgs/PointCloud2` messages
* `target_frame`: name of reference frame which the object poses will be transformed to
* `class_annotations`: YAML file which maps numeric class values to class names, used by the
[`ImageDetectorBase` class](docs/python_package.md) to configure the detection model. An example of this file is
[`class_annotation_example.yml`](models/class_annotation_example.yml).
* `kwargs_file`: YAML file which is used by the [`ImageDetectorBase` class](docs/python_package.md) to specify
additional parameters needed to configure the detection model. An example is
[`image_detector_test_kwargs.yml`](models/image_detector_test_kwargs.yml).
* `detection_module`: name of the module containing the `ImageDetectorBase` extension to import.
* `detection_class`: name of the extension of the `ImageDetectorBase` class to import.
* also parameters defined in the dynamic reconfiguration file [PlaneFitting.cfg](ros/config/PlaneFitting.cfg).

### [`image_detection_test`](ros/scripts/image_detection_test)
Node for testing image detection models. Can test images from a directory, a `sensor_msgs/Image` topic, or a
`sensor_msgs/PointCloud2` topic.

Parameters:
* `class_annotations`, `kwargs_file`, `detection_module`, and `detection_class`: parameters for `ImageDetectorBase`
class similar to ones described above for `object_detection_action_server` and `plane_detection_action_server`.
* `result_topic`: `sensor_msgs/Image` topic which visualized detection results are published.
* `image_directory`: if specified will ignore other image sources and read images from this directory for testing.
* `cloud_topic`: if specified and `image_directory` is not specified will extract images from `sensor_msgs/PointCloud2`
messages from this topic for testing.
* `image_topic`: if specified and the other 2 image sources are not specified will test detection model on
`sensor_msgs/Image` messages from this topic.

### [`image_recognition_server`](ros/scripts/image_recognition_server)
Server which uses an instance of `ImageClassifier` class (See
[documentation](docs/python_package.md)) to classify images.

Parameters:
* `service_name`: name of recognition service to advertise (default: `'~recognize_image'`).
* `model_directory`: directory containing the trained classification model (default: `''`)
* `recognition_module`: module containing the `ImageClassifier` instance
(default: `'mas_perception_libs'`)
* `recognition_class`: class name of the `ImageClassifier` instance
(default: `'ImageClassifierTest'`)

### [`image_recognition_client_test`](ros/scripts/image_recognition_client_test)
Script to test the `image_recognition_server`.

```
usage: image_recognition_client_test [-h] --test-dir TEST_DIR --service-name
                                     SERVICE_NAME [--num-samples NUM_SAMPLES]
                                     [--preprocess-input-module PREPROCESS_INPUT_MODULE]
                                     model_name

Tool to test model with test images using KerasImageClassifier class.Assuming
images to be of type jpg

positional arguments:
  model_name            Keras model to be tested

optional arguments:
  -h, --help            show this help message and exit
  --test-dir TEST_DIR, -t TEST_DIR
                        directory with test images
  --service-name SERVICE_NAME, -s SERVICE_NAME
                        name of recognition service
  --num-samples NUM_SAMPLES, -n NUM_SAMPLES
                        number of samples to test, if left blank, take all
                        samples.
  --preprocess-input-module PREPROCESS_INPUT_MODULE, -p PREPROCESS_INPUT_MODULE
                        module containing image preprocessing function.
```

Example execution:
```
image_recognition_client_test -s <service_name> -t <folder_with_test_images> <model_name>
```

### [`cloud_processing_python_test`](ros/scripts/cloud_processing_python_test)
Script for testing `PlaneSegmenter` (see [Python documentation](docs/python_package.md)), including handling of the
dynamic reconfiguration defined in [PlaneFitting.cfg](ros/config/PlaneFitting.cfg). Parameters:
* `cloud_topic` (`string`): input point cloud topic
* `processed_cloud_topic` (`string`): topic on which the filtered cloud will be published
* `target_frame` (`string`): coordinate frame that the point cloud will be transformed to.
* `extract_planes`(`bool`): if `false` will only do cloud filtering
* parameters defined in [PlaneFitting.cfg](ros/config/PlaneFitting.cfg)

### [`cloud_processing_cpp_test`](ros/src/cloud_processing_test_node.cpp)
C++ executable testing `PlaneSegmenterROS` (see [C++ documentation](docs/cpp_library.md)), including handling of the
dynamic reconfiguration defined in [PlaneFitting.cfg](ros/config/PlaneFitting.cfg). Parameters:
* `cloud_topic` (`string`): input point cloud topic
* `processed_cloud_topic` (`string`): topic on which the filtered cloud will be published
* `target_frame` (`string`): coordinate frame that the point cloud will be transformed to.
* `extract_planes`(`bool`): if `false` will only do cloud filtering
* parameters defined in [PlaneFitting.cfg](ros/config/PlaneFitting.cfg)

## Launch Files

### [`plane_detection.launch`](ros/launch/plane_detection.launch)
Launch the `plane_detection_action_server` script. Arguments are the same with the script's parameters:
* `action_name` (default: `"/mas_perception/detect_image"`)
* `cloud_topic` (default: `""`)
* `target_frame` (default: `"/base_link"`)
* `class_annotations` (default: `"$(find mas_perception_libs)/models/class_annotation_example.yml"`)
* `kwargs_file` (default: `"$(find mas_perception_libs)/models/image_detector_test_kwargs.yml"`)
* `detection_module` (default: `"mas_perception_libs"`)
* `detection_class` (default: `"ImageDetectorTest"`)
* `plane_fitting_config_file` (default `"$(find mas_perception_libs)/ros/config/plane_fitting_default_configs.yaml"`)

### [`object_detection.launch`](ros/launch/object_detection.launch)
Launch the `object_detection_action_server` script. Arguments are the same with the script's parameters:
* `action_name` (default: `"/mas_perception/detect_image"`)
* `cloud_topic` (default: `""`)
* `target_frame` (default: `"/base_link"`)
* `class_annotations` (default: `"$(find mas_perception_libs)/models/class_annotation_example.yml"`)
* `kwargs_file` (default: `"$(find mas_perception_libs)/models/image_detector_test_kwargs.yml"`)
* `detection_module` (default: `"mas_perception_libs"`)
* `detection_class` (default: `"ImageDetectorTest"`)

### [`image_detection_test.launch`](ros/launch/image_detection_test.launch)
Launch the `image_detection_test` script. Arguments are the same with the script's parameters:
* `result_topic` (default: `"/mas_perception/detection_result"`)
* `image_directory` (default: `""`)
* `cloud_topic` (default: `""`)
* `image_topic` (default: `""`)
* `class_annotations` (default: `"$(find mas_perception_libs)/models/class_annotation_example.yml"`)
* `kwargs_file` (default: `"$(find mas_perception_libs)/models/image_detector_test_kwargs.yml"`)
* `detection_module` (default: `"mas_perception_libs"`)
* `detection_class` (default: `"ImageDetectorTest"`)

### [`image_recognition.launch`](ros/launch/image_recognition.launch)
Launch the `image_recognition_server`. Arguments:
* `service_name`: name of recognition service to advertise (default: `'~recognize_image'`).
* `model_directory`: directory containing the trained classification model
(default: `$(find mas_perception_libs)/models`)
* `recognition_module`: module containing the `ImageClassifier` instance
(default: `'mas_perception_libs'`)
* `recognition_class`: class name of the `ImageClassifier` instance
(default: `'ImageClassifierTest'`)

### [`cloud_processing_python_test.launch`](ros/launch/cloud_processing_python_test.launch)
Launch the [`cloud_processing_python_test`](ros/scripts/cloud_processing_python_test) executable. Arguments are similar
to ones for the executable, with an additional `plane_fitting_config_file` in which the default values for the dynamic
reconfiguration [PlaneFitting.cfg](ros/config/PlaneFitting.cfg). Default file is
[`plane_fitting_default_configs.yaml`](ros/config/plane_fitting_default_configs.yaml).

### [`cloud_processing_cpp_test.launch`](ros/launch/cloud_processing_cpp_test.launch)
Launch the [`cloud_processing_cpp_test`](ros/src/cloud_processing_test_node.cpp) executable. Arguments are similar
to ones for the executable, with an additional `plane_fitting_config_file` in which the default values for the dynamic
reconfiguration [PlaneFitting.cfg](ros/config/PlaneFitting.cfg). Default file is
[`plane_fitting_default_configs.yaml`](ros/config/plane_fitting_default_configs.yaml).

## Directory structure

```
.
├── CMakeLists.txt
├── common
│   ├── include
│   │   └── mas_perception_libs
│   │       ├── aliases.h
│   │       ├── bounding_box_2d.h
│   │       ├── bounding_box.h
│   │       ├── impl
│   │       │   └── pyboostcvconverter.hpp
│   │       ├── point_cloud_utils.h
│   │       ├── sac_plane_segmenter.h
│   │       └── use_numpy.h
│   └── src
│       ├── bounding_box_2d.cpp
│       ├── bounding_box.cpp
│       ├── init_numpy_api.cpp
│       ├── point_cloud_utils.cpp
│       ├── pyboost_cv3_converter.cpp
│       └── sac_plane_segmenter.cpp
├── docs
│   ├── cpp_library.md
│   ├── image_detection_class_structure.png
│   └── python_package.md
├── models
│   ├── class_annotation_example.yml
│   ├── image_detector_test_kwargs.yml
│   └── test_model.txt
├── package.xml
├── README.md
├── ros
│   ├── config
│   │   ├── PlaneFitting.cfg
│   │   └── plane_fitting_default_configs.yaml
│   ├── include
│   │   └── mas_perception_libs
│   │       ├── bounding_box_wrapper.h
│   │       ├── color.h
│   │       ├── image_bounding_box.h
│   │       ├── impl
│   │       │   └── ros_message_serialization.hpp
│   │       └── point_cloud_utils_ros.h
│   ├── launch
│   │   ├── cloud_processing_cpp_test.launch
│   │   ├── cloud_processing_python_test.launch
│   │   ├── image_detection.launch
│   │   ├── object_detection.launch
│   │   ├── image_detection_test.launch
│   │   └── image_recognition.launch
│   ├── scripts
│   │   ├── cloud_processing_python_test
│   │   ├── plane_detection_action_server
│   │   ├── object_detection_action_server
│   │   ├── image_detection_test
│   │   ├── image_recognition_client_test
│   │   └── image_recognition_server
│   └── src
│       ├── boost_python_module.cpp
│       ├── bounding_box_wrapper.cpp
│       ├── cloud_processing_test_node.cpp
│       ├── image_bounding_box.cpp
│       ├── mas_perception_libs
│       │   ├── bounding_box.py
│       │   ├── constants.py
│       │   ├── image_classifier.py
│       │   ├── image_detector.py
│       │   ├── image_recognition_service.py
│       │   ├── __init__.py
│       │   ├── object_detector.py
│       │   ├── ros_message_serialization.py
│       │   ├── scene_detection_action.py
│       │   ├── utils.py
│       │   └── visualization.py
│       └── point_cloud_utils_ros.cpp
└── setup.py
```
