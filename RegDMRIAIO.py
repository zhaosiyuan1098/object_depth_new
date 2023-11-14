# Copyright (C) USTC BMEC RFLab - All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# \Author Qaler Qi, \date created on 2023-04-19
from pickle import TRUE
import numpy as np
from RefMarkerReg import MarkerReg
from MRPtRegPack import PtRecon, MRReg
from RSRawParser import offline_acq
import open3d as o3d
from rsworker import RSWorker
from headReg import reg_head

if __name__ == "__main__":
    folder = "data/regtest/0804_3"
    folderface = folder +'/face'
    
    # \TODO: Extract image and depth frames from .bag file  
    offline_acq(folder, folder + "/20230804_195916.bag", show_plt=True)

    # Detect ref and calibration markers orientation/position
    # in camera coordinate.
    mReg = MarkerReg(folder)
    mReg.reg_marker()
    
    # Detect calibration phantom orientation/position in
    # MRI coordinate.
    ptR = PtRecon(folder + "/lineK/")
    ptR.ask_recon_loop()
    
    # Combine the previous two steps results, calculate
    # final transformation between camera and MRI, output
    # camToMRI and refToMRI.
    mReg = MRReg(folder)
    mReg.reg_calib()


    # 面部点云配准部分
    offline_acq(folderface, folderface + "/20230804_200221.bag", show_plt=True)
    mReg = MarkerReg(folderface)
    mReg.reg_marker()
    pcd = o3d.io.read_point_cloud(folderface+'/scene.pcd')
    o3d.io.write_point_cloud(folderface + '/head_piece' + ".ply", pcd)
    RSWorker.DbScanClustering(folderface,
        pcd, numface=0, eps=5, min_points=100 )
    # 运行geometric_feature，配准过程
    reg_head(folder,folderface)



