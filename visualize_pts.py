import numpy as np
import open3d as o3d

def visualize(filename):
    cloud = o3d.io.read_point_cloud(filename) # Read the point cloud
    o3d.visualization.draw_geometries([cloud]) # Visualize the point cloud
visualize('align.pcd')