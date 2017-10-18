# Project: Perception Pick & Place

---


# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!


[//]: # (Image References)

[first]: ./images/overall_scene.png
[image0]: ./images/raw_points.png
[image1]: ./images/voxel.png
[image2]: ./images/passthrough.png
[image3]: ./images/outlier.png
[image4]: ./images/RANSAC.png
[image5]: ./images/cluster1.png
[image6]: ./images/cluster2.png
[image7]: ./images/model.png
[image8]: ./images/world2.png
[image9]: ./images/world1.png
[image10]: ./images/world3.png

![Overall Scene][first]

### Exercise 1, 2 and 3 Implemented
#### Filtering and RANSAC Plane Fitting
The raw point cloud detected by the RGBD camera is shown below.
![Raw Points][image0]

The first step in the pipeline was to sample these down into smaller voxels to reduce the amount of computation required. Next a passthrough filter in both height and forward direction was implemented to crop out unneccessary parts of the scene.
![Passthrough][image2]

Outliers were removed using from the point cloud.
![Statistical Outlier Removal][image3]

RANSAC plane fitting was used to detect and removed the table.
![RANSAC][image4]

#### Clustering
The points belonging to the same objects are then grouped together using euclidean clustering.
![Original][image5]
![Clustered][image6] 

#### SVM Object Recognition
SVMs were trained on point clouds of the objects rotated randomly to get a view from various different angles. The features used in the SVM was a concatenation of HSV value histograms along with spatial location histogram.

The results for the model trained are shown below.

![Model][image7]
![Prediction][image8]

For the first world I was able to detect all 3 objects correctly. For the second world (shown above) I was only able to detect 4/5 objects correctly (glue is mislabeled as biscuit). This is odd since in the model results when the true label was glue, biscuits was never guessed once. For the third world the SVM detected 6/8 objects correctly. For this last one, glue was mislabeled as sticky notes, and snacks wasn't detected at all. 

These results are shown below.
![World 1][image9]
![World 3][image10]