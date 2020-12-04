# import numpy as np
import open3d as o3d

def visualize(filename):
    cloud = o3d.io.read_point_cloud(filename) # Read the point cloud
    o3d.geometry.estimate_normals(
        cloud,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.8,
                                                          max_nn=30))
    o3d.visualization.draw_geometries([cloud]) # Visualize the point cloud

visualize('higher3.pcd')
