import numpy as np
import open3d as o3d
from pytransform3d import transformations as pt
from skspatial.objects import Plane
import sigpy.plot as pl
import trimesh
import numbers
import SimpleITK as sitk
import time
import math
import copy
import os

head = o3d.io.read_triangle_mesh(r'data\trans1219.ply')
fortrans1 = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 38],
                      [0,0,0,1]])
head.transform(fortrans1)
o3d.io.write_triangle_mesh(r'data\trans1220' + '.ply', head)
