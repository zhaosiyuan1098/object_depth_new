# Copyright (C) USTC BMEC RFLab - All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# \Author Qaler Qi, \date created on 2022-07-12
import numpy as np
import cv2
import open3d as o3d
from PIL import Image, ImageDraw, ImageEnhance
from scipy.linalg import orthogonal_procrustes
from skspatial.objects import Plane
from pytransform3d import transformations as pytr
from pytransform3d.transform_manager import TransformManager
import sigpy.plot as pl
import json
import utils
import copy
import random

class MarkerReg(object):
    def __init__(self, folder, show_plt=False, debug=True) -> None:
        super().__init__()
        self.folder = folder
        self.show_plt = show_plt
        self.debug = debug
        self.configJ = {}
        self.update_config()

    def update_config(self):
        configF = self.folder + "/markerReg.json"
        try:
            with open(configF, "r", encoding="utf-8") as f:
                self.configJ = json.load(f)
            if self.configJ is not None:
                if "showPlt" in self.configJ:
                    self.show_plt = self.configJ.get("showPlt", self.show_plt)
                if "debug" in self.configJ:
                    self.debug = self.configJ.get("debug", self.debug)
        except Exception as e:
            print("Update config fails:", e)

    @staticmethod
    def enhance_light(im, factor=3):
        im = Image.fromarray(im)
        enhancer = ImageEnhance.Brightness(im)
        im_output = enhancer.enhance(factor)
        contrast_enhancer = ImageEnhance.Contrast(im_output)
        im_output = contrast_enhancer.enhance(factor)
        return np.asarray(im_output)

    @staticmethod
    def detect_markers(
        infra, aruco_dict=None, aruco_params=None, debug=False, enhance_factor=3
    ):
        if aruco_dict is None:
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
        if aruco_params is None:
            aruco_params = cv2.aruco.DetectorParameters_create()
        if infra.ndim == 2:
            infra_3d = np.dstack((infra, infra, infra))
        else:
            infra_3d = infra
        corners, ids, rejected = cv2.aruco.detectMarkers(
            MarkerReg.enhance_light(infra_3d, enhance_factor)
            if enhance_factor is not None
            else infra_3d,
            aruco_dict,
            parameters=aruco_params,
        )
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(infra_3d, corners, ids)
            if debug:
                print("Detected markers: ", ids)
        return corners, ids, rejected, infra_3d

    @staticmethod
    def create_mask(height, width, corner):
        # https://stackoverflow.com/questions/49676926/why-pil-draw-polygon-does-not-accept-numpy-array
        corner = np.squeeze(corner)
        img = Image.new("L", (width, height), 0)
        ImageDraw.Draw(img).polygon(
            corner.flatten().tolist(), outline=1, fill=1)
        return np.array(img).copy()

    @staticmethod
    def show_seg_plane(pcd, inliers):
        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    @staticmethod
    def get_pts_3d(pts, depth_frame, pinhole_intrinsic):
        # TODO: use an average vector along pts between
        # origin and [dx, dy], instead of only using v(o, dx) and v(o, dy)
        pts = np.squeeze(pts)
        mask = np.zeros_like(depth_frame)
        pts_3d = []
        for pt_2d in pts:
            pt_2d = pt_2d.astype(int)
            mask_i = mask.copy()

            # NOTE: not sure why reversed is needed
            mask_i[tuple(reversed(pt_2d.astype(int)))] = 1

            d_p = depth_frame * mask_i
            if False:
                # FIXME: doesn't produce correct points
                pt_3d = utils.convert_depth_frame_to_pointcloud(
                    d_p, pinhole_intrinsic)
                pt_3d = np.asarray(pt_3d) * 1000
                pts_3d.append([pt_3d[0][0], pt_3d[1][0], pt_3d[2][0]])
            else:
                pcd = utils.depthToPcd(d_p, pinhole_intrinsic)
                pts_3d.append(np.squeeze(np.asarray(pcd.points)))
        return pts_3d

    @staticmethod
    def get_marker_dir_2d(corner):
        corner = np.squeeze(np.asarray(corner))
        if np.shape(corner)[0] < 4:
            raise ValueError(
                "marker must have 4 corner points:", np.shape(corner))
        line_centers = []
        for i in range(4):
            line_centers.append((corner[i] + corner[(i + 1) % 4]) / 2)
        center = np.mean(corner, axis=0)
        dx = center + (line_centers[1] - center) / 2
        dy = center + (line_centers[0] - center) / 2
        return center, dx, dy

    @staticmethod
    def get_plane_obj(plane_model, debug=True):
        # a*x + b*y + c*z + d = 0
        [a, b, c, d] = plane_model
        if debug:
            print(
                f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0.")
        pt = [0, 0, 0]
        if abs(a) > 1e-3:
            pt[0] = -d / a
        elif abs(b) > 1e-3:
            pt[1] = -d / b
        elif abs(c) > 1e-3:
            pt[2] = -d / c
        return Plane(pt, [a, b, c])

    @staticmethod
    def marker_pcd(
        depth_frame, color_frame, mask, pinhole_intrinsic, clipping_distance=1000
    ):
        mask_3d = np.dstack((mask, mask, mask))
        pcd = utils.rgbdToPcd(
            depth_frame * mask, color_frame * mask_3d, pinhole_intrinsic
        )
        if clipping_distance > 0:
            # reserve points inside distance (mm) from camera
            distance = np.linalg.norm(np.asarray(pcd.points), axis=1)
            pcd = pcd.select_by_index(
                np.where(distance < clipping_distance)[0])
        return pcd

    @staticmethod
    def draw_dir(im_np, center, dx, dy, show=False):
        im = Image.fromarray(im_np)
        draw = ImageDraw.Draw(im)
        draw.line([tuple(center), tuple(dx)], fill=(255, 255, 0))  # yellow
        draw.line([tuple(center), tuple(dy)], fill=(0, 255, 255))  # lime
        if show:
            pl.ImagePlot(im_np, x=0, y=1, c=2)
        return np.array(im)

    @staticmethod
    def get_marker_coord(center, px, py, normal, debug=True):
        vx = px - center
        vx = vx / np.linalg.norm(vx)
        vy = py - center
        vy = vy / np.linalg.norm(vy)
        if debug:
            print(
                "normal",
                normal,
                ". dot(vx, normal)",
                np.dot(vx, normal),
                ". dot(vy, normal)",
                np.dot(vy, normal),
            )
        marker_pts = np.asarray([vx, vy, normal])
        origin = np.identity(3)

        # If the correspondences are known, the solution to the
        # rigid registration is known as the orthogonal Procrustes problem.
        # NOTE vx is not exactly perpendicular to vy due to system errors.
        # NOTE for coord transformation, it is marker coord to origin coord,
        # that is camera coord to phantom coord, in passive mode.
        # For active mode though, it is phantom to camera.
        # Check `reg_calib` in `CalibPtsReg` for detailed explanation.
        R, sca = orthogonal_procrustes(origin, marker_pts)

        if np.linalg.det(R) < 0:
            # if rotation matrix determinant is negative, then the
            # normal vector is probably left-hand sided to the vx, vy.
            # So we'll need to reverse normal direction.
            print("inverse plane normal")
            marker_pts = np.asarray([vx, vy, -1 * normal])
            R, sca = orthogonal_procrustes(origin, marker_pts)

        rmse = np.linalg.norm(origin @ R - marker_pts) / \
            np.sqrt(len(marker_pts))
        if debug:
            print("marker_pts", marker_pts)
            print("origin @ R", origin @ R)
            print("marker coord reg sca:", sca, ". rmse", rmse)
        return R, center, sca

    @staticmethod
    def plantmodel(plant, x, y, z):
        a, b, c, d = plant
        t = (a*x + b*y + c*z + d) / (a*a + b*b + c*c)
        return [x - a*t, y - b*t, z - c*t]
    
    def get_marker_plane_manually(
        self,
        depth_frame,
        infra_frame,
        pinhole_camera_intrinsic,
        corner,
        id,
        show_plt=True,
        debug=True,
    ):
        """In case marker could not be detected automatically,
        we can manually input the approximate corner location.
        """
        img_shape = np.shape(infra_frame)
        center, dx, dy = MarkerReg.get_marker_dir_2d(corner)

        if infra_frame.ndim == 2:
            infra_3d_dir = np.dstack((infra_frame, infra_frame, infra_frame))
        else:
            infra_3d_dir = infra_frame
        infra_3d_dir = MarkerReg.draw_dir(infra_3d_dir, center, dx, dy)

        dirs_3d = MarkerReg.get_pts_3d(
            [center, dx, dy], depth_frame, pinhole_camera_intrinsic
        )
        mask = MarkerReg.create_mask(img_shape[0], img_shape[1], corner)

        pcd = MarkerReg.marker_pcd(
            depth_frame, infra_3d_dir.copy(), mask, pinhole_camera_intrinsic, -1
        )
        print("marker pcd center in cam", pcd.get_center())
        o3d.visualization.draw_geometries([pcd])
        # np.random.seed(0)
        # random.seed(0)
        # o3d.utility.random.seed(0)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=5, ransac_n=3, num_iterations=1000
        )
        if show_plt:
            MarkerReg.show_seg_plane(pcd, inliers)
            
        plane = MarkerReg.get_plane_obj(plane_model, debug=debug)
        print('before project_point', dirs_3d)
        dirs_3d_plane = []
        try:
            for pt_3d in dirs_3d:
                dirs_3d_plane.append(plane.project_point(pt_3d))
            if True or debug:
                print(id, "dirs_3d_plane", dirs_3d_plane)
        except ValueError as e:
            if show_plt:
                print(e)

        if len(dirs_3d_plane) < 3:
            return None, None

        # ref marker dirs_3d_plane [Point([ -34.9669652 , -179.18662138, -722.45765868]),
        #       Point([ -53.92288434, -178.77492151, -720.18840699]),
        #       Point([ -34.79545177, -196.74995543, -717.40922335])]
        R, translation, _ = MarkerReg.get_marker_coord(
            dirs_3d_plane[0],
            dirs_3d_plane[1],
            dirs_3d_plane[2],
            plane.normal,
            debug=debug,
        )

        planeJ = {}
        planeJ["id"] = id
        # NOTE: for reason of R.T, check reg_all_feasible test in CalibPtsReg.py
        # matrix is conversion of
        planeJ["matrix"] = pytr.transform_from(R.T, translation.T)
        print("id", id, " pcd center:", pcd.get_center())
        return planeJ, pcd

    def marker_to_plane(
        self,
        depth_frame,
        infra_frame,
        pinhole_camera_intrinsic,
    ):
        enhance_factor = self.configJ.get("enhanceFactor", 3)
        clip_range = self.configJ.get("clipRange", 800)

        corners, ids, _, infra_3d = MarkerReg.detect_markers(
            infra_frame, enhance_factor=enhance_factor
        )
        if self.show_plt:
            pl.ImagePlot(infra_3d, y=0, x=1, c=2)

        if ids is None:
            return None, None

        if self.show_plt:
            pcd_all = MarkerReg.marker_pcd(
                depth_frame,
                infra_3d.copy(),
                np.ones_like(depth_frame, dtype=np.uint8),
                pinhole_camera_intrinsic,
                clip_range,
            )
            o3d.visualization.draw_geometries([pcd_all])

        planes = []
        infra_3d_dir = np.copy(infra_3d)
        for corner, id in zip(corners, ids):
            planeJ, pcd = self.get_marker_plane_manually(
                depth_frame,
                infra_frame,
                pinhole_camera_intrinsic,
                corner,
                id,
                self.show_plt,
                self.debug,
            )
            if planeJ is None:
                continue
            planes.append(copy.deepcopy(planeJ))

            if self.show_plt:
                o3d.visualization.draw_geometries([pcd])

            if self.folder is not None:
                o3d.io.write_point_cloud(
                    self.folder + "/marker_" + str(int(id)) + ".pcd", pcd
                )

        if self.folder is not None:
            pcd_all = MarkerReg.marker_pcd(
                depth_frame,
                infra_3d_dir.copy(),
                np.ones_like(depth_frame, dtype=np.uint8),
                pinhole_camera_intrinsic,
                clip_range,
            )
            o3d.io.write_point_cloud(self.folder + "/scene.pcd", pcd_all)

        if self.folder is not None:
            Image.fromarray(infra_3d_dir).save(self.folder + "/scene.png")

        return planes, infra_3d_dir

    def reg_marker(
        self,
        detect_all=None,
        frame_idx=None,
        ref_marker_idx=None,
        calib_marker_idx=None,
    ):
        """Register calibration object and ref marker coord to depth camera system.

        Args:
            detect_all (boolean, optional): If to show detected results for all images. Defaults to False.
                This option is usually used for determine a valid frame_idx.
            frame_idx (int, optional): _description_. Defaults to 11.
            ref_marker_idx (int, optional): _description_. Defaults to 15.
            calib_marker_idx (int, optional): _description_. Defaults to 17.

        Returns:
            _type_: TransformManager
        """
        self.update_config()
        if detect_all is None:
            detect_all = self.configJ.get("detectAll", False)
        if frame_idx is None:
            frame_idx = self.configJ.get("frameIdx", 11)
        if ref_marker_idx is None:
            ref_marker_idx = self.configJ.get("refMarkerIdx", 15)
        if calib_marker_idx is None:
            calib_marker_idx = self.configJ.get("calibMarkerIdx", 8)

            

        print(
            f"Detect all {detect_all}, use frame {frame_idx}, "
            f"marker idx {ref_marker_idx}, calib idx {calib_marker_idx}"
        )

        depth_frame_list = np.load(self.folder + "/depthFrameList.npy")
        infra_frame_list = np.load(self.folder + "/infraFrameList.npy")
        print(
            f"Depth frames {len(depth_frame_list)}, infra frames {len(infra_frame_list)}"
        )

        with open(self.folder + "/intrinsics.json", "r") as ff:
            intrinsics = json.load(ff)
        print("intrinsics", intrinsics)
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["fx"],
            intrinsics["fy"],
            intrinsics["ppx"],
            intrinsics["ppy"],
        )

        if detect_all:
            ids_all = []
            infra_3d_detected_list = []
            for infra in infra_frame_list:
                corners, ids, _, infra_3d = MarkerReg.detect_markers(infra)
                if ids is not None:
                    ids_all.append(ids)
                else:
                    ids_all.append([])
                infra_3d_detected_list.append(np.asarray(infra_3d).copy())
            if self.show_plt:
                pl.ImagePlot(
                    np.asarray(infra_3d_detected_list),
                    z=0,
                    y=1,
                    x=2,
                    c=3,
                    title="infra_frame_list",
                    showPlt=True,
                )
            selected_ids = ids_all[frame_idx]
            if ref_marker_idx in selected_ids and calib_marker_idx in selected_ids:
                print("frame_idx is valid: ", frame_idx, selected_ids)
            else:
                print("Warning: frame_idx is invalid: ",
                      frame_idx, selected_ids)
                frame_idx = None
                for idx, ids in enumerate(ids_all):
                    if ref_marker_idx in ids and calib_marker_idx in ids:
                        print("Change frame_idx to: ", idx, ids)
                        frame_idx = idx
                        break

        if frame_idx is None:
            raise ValueError(
                "No valid frames that contains: ", ref_marker_idx, calib_marker_idx
            )

        depth_frame = depth_frame_list[frame_idx]
        infra_frame = infra_frame_list[frame_idx]

        planes, _ = self.marker_to_plane(
            depth_frame, infra_frame, pinhole_camera_intrinsic
        )
        for planeJ in planes:
            id = planeJ["id"]
            # NOTE matrix is in passive mode
            if id == ref_marker_idx:
                cam_to_ref_p = planeJ["matrix"]
            elif id == calib_marker_idx:
                cam_to_calib_p = planeJ["matrix"]
        np.save(self.folder + "/cam_to_ref_passive.npy", cam_to_ref_p)
        if len(planes) < 2:
            print("could not find both ref and calib markers.")
            return None
        
        diff_vec = cam_to_calib_p[:3, 3] - cam_to_ref_p[:3, 3]
        print('diff_vec', np.linalg.norm(diff_vec))

        # np.save(self.folder + "/cam_to_ref_passive.npy", cam_to_ref_p)
        np.save(self.folder + "/cam_to_calib_passive.npy", cam_to_calib_p)
        tm_p = TransformManager()
        tm_p.add_transform("ref", "cam", cam_to_ref_p)
        tm_p.add_transform("calib", "cam", cam_to_calib_p)
        print('ref to calib', tm_p.get_transform("ref", "calib"))
        return tm_p


if __name__ == "__main__":
    folder = "data/regtest/0606/"
    mReg = MarkerReg(folder)
    mReg.reg_marker()
