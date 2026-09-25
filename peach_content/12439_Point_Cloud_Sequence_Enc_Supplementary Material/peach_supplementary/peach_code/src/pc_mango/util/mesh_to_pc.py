import math
import time
from typing import Dict

import numpy as np
import open3d as o3d
from tqdm import tqdm
import fpsample

VOXEL_SIZE = 0.005


def convert_to_point_cloud_from_mask(ans, rays, mask):
    t_hit = ans["t_hit"].numpy()
    rays_np = rays.numpy()
    points = rays_np[mask, :3] + rays_np[mask, 3:] * t_hit[mask, np.newaxis]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def mesh_to_pcd_with_occlusions(dataset_name, mesh_pos, mesh_faces, collider_pos=None, collider_faces=None,
                                num_final_points=2000, cropping=None, visualize=False):
    t0 = time.time()

    if mesh_pos.shape[-1] == 2:
        mesh_pos = np.concatenate([mesh_pos, np.zeros((len(mesh_pos), 1))], axis=1)
    if collider_pos is not None and collider_pos.shape[-1] == 2:
        collider_pos = np.concatenate([collider_pos, np.zeros((len(collider_pos), 1))], axis=1)

    vertices_tensor = o3d.core.Tensor(mesh_pos, dtype=o3d.core.Dtype.Float32)
    faces_tensor = o3d.core.Tensor(mesh_faces, dtype=o3d.core.Dtype.UInt32)

    has_collider = collider_pos is not None
    camera_eyes, ups, center = get_camera_eyes(dataset_name)
    fov_deg = 70 # high fov to ensure we capture everything
    t1 = time.time()

    if has_collider:
        assert collider_faces is not None
        v2 = o3d.core.Tensor(collider_pos, dtype=o3d.core.Dtype.Float32)
        f2 = o3d.core.Tensor(collider_faces, dtype=o3d.core.Dtype.UInt32)

        # Single combined scene so each object occludes the other
        combined_scene = o3d.t.geometry.RaycastingScene()
        mesh_geom_id = combined_scene.add_triangles(vertices_tensor, faces_tensor)
        collider_geom_id = combined_scene.add_triangles(v2, f2)

        mesh_pcd_sum = o3d.t.geometry.PointCloud().to_legacy()
        collider_pcd_sum = o3d.t.geometry.PointCloud().to_legacy()

        for camera_eye, up in zip(camera_eyes, ups):
            ans, rays = do_raycasting(combined_scene, fov_deg=fov_deg, center=center,
                                      eye=camera_eye, up=up)
            hit_geom = ans["geometry_ids"].numpy()
            hit = ans["t_hit"].numpy() < np.inf
            mesh_pcd_sum += convert_to_point_cloud_from_mask(ans, rays, hit & (hit_geom == mesh_geom_id))
            collider_pcd_sum += convert_to_point_cloud_from_mask(ans, rays, hit & (hit_geom == collider_geom_id))

        t2 = time.time()

        mesh_points = np.asarray(mesh_pcd_sum.voxel_down_sample(VOXEL_SIZE).points)
        collider_points = np.asarray(collider_pcd_sum.voxel_down_sample(VOXEL_SIZE).points)

        t3 = time.time()

        pcd_combined = np.concatenate([mesh_points, collider_points], axis=0)
        node_types = np.concatenate([
            np.zeros(len(mesh_points), dtype=np.int32),
            np.ones(len(collider_points), dtype=np.int32),
        ])
        if cropping is not None:
            mask = (
                    (pcd_combined[:, 0] >= cropping["x_min"]) & (pcd_combined[:, 0] <= cropping["x_max"]) &
                    (pcd_combined[:, 1] >= cropping["y_min"]) & (pcd_combined[:, 1] <= cropping["y_max"]) &
                    (pcd_combined[:, 2] >= cropping["z_min"]) & (pcd_combined[:, 2] <= cropping["z_max"])
            )
            pcd_combined = pcd_combined[mask]
            node_types = node_types[mask]

        fps_indices = fpsample.bucket_fps_kdtree_sampling(pcd_combined, num_final_points)
        fps_subsampled = pcd_combined[fps_indices]
        node_types_subsampled = node_types[fps_indices]

    else:
        # No collider — single mesh scene, all nodes are type 0
        mesh_scene = o3d.t.geometry.RaycastingScene()
        mesh_scene.add_triangles(vertices_tensor, faces_tensor)

        pcd_sum = o3d.t.geometry.PointCloud().to_legacy()
        for camera_eye, up in zip(camera_eyes, ups):
            ans, rays = do_raycasting(mesh_scene, fov_deg=fov_deg, center=center,
                                      eye=camera_eye, up=up)
            pcd_sum += convert_to_point_cloud(ans, rays)

        t2 = time.time()
        pcd_subsampled = np.asarray(pcd_sum.voxel_down_sample(VOXEL_SIZE).points)
        t3 = time.time()

        fps_indices = fpsample.bucket_fps_kdtree_sampling(pcd_subsampled, num_final_points)
        fps_subsampled = pcd_subsampled[fps_indices]
        node_types_subsampled = np.zeros(len(fps_subsampled), dtype=np.int32)

    t4 = time.time()
    if visualize:
        print(f"init: {t1 - t0:.3f}s")
        print(f"raycasting: {t2 - t1:.3f}s")
        print(f"voxel downsample: {t3 - t2:.3f}s")
        print(f"fps: {t4 - t3:.3f}s")
        vis_pcd(fps_subsampled, camera_eyes, None, node_types_subsampled)

    return fps_subsampled, node_types_subsampled


def mesh_to_pcd(dataset_name, mesh_pos, mesh_faces, collider_pos=None, collider_faces=None, num_final_points=2000,
                visualize=False, ):
    # Create a TriangleMesh
    # tissue_mesh = o3d.geometry.TriangleMesh()
    t0 = time.time()
    # Convert numpy arrays to Open3D tensors
    # add third dim if needed
    if mesh_pos.shape[-1] == 2:
        mesh_pos = np.concatenate([mesh_pos, np.zeros((len(mesh_pos), 1))], axis=1)
    if collider_pos is not None and collider_pos.shape[-1] == 2:
        collider_pos = np.concatenate([collider_pos, np.zeros((len(collider_pos), 1))], axis=1)

    # Create mesh scene
    vertices_tensor = o3d.core.Tensor(mesh_pos, dtype=o3d.core.Dtype.Float32)
    faces_tensor = o3d.core.Tensor(mesh_faces, dtype=o3d.core.Dtype.UInt32)
    mesh_scene = o3d.t.geometry.RaycastingScene()
    mesh_scene.add_triangles(vertices_tensor, faces_tensor)

    has_collider = collider_pos is not None
    if has_collider:
        assert collider_faces is not None
        # Create separate collider scene
        v2 = o3d.core.Tensor(collider_pos, dtype=o3d.core.Dtype.Float32)
        f2 = o3d.core.Tensor(collider_faces, dtype=o3d.core.Dtype.UInt32)
        collider_scene = o3d.t.geometry.RaycastingScene()
        collider_scene.add_triangles(v2, f2)

    # generate point cloud
    camera_eyes, ups, center = get_camera_eyes(dataset_name)
    t1 = time.time()

    if has_collider:
        # Separate point clouds for mesh and collider using separate scenes
        mesh_pcd_sum = o3d.t.geometry.PointCloud().to_legacy()
        collider_pcd_sum = o3d.t.geometry.PointCloud().to_legacy()

        for camera_eye, up in zip(camera_eyes, ups):
            # Raycast mesh scene
            ans_mesh, rays_mesh = do_raycasting(mesh_scene, fov_deg=42, center=center, eye=camera_eye, up=up)
            mesh_pcd = convert_to_point_cloud(ans_mesh, rays_mesh)
            mesh_pcd_sum += mesh_pcd

            # Raycast collider scene
            ans_collider, rays_collider = do_raycasting(collider_scene, fov_deg=42, center=center,
                                                        eye=camera_eye, up=up)
            collider_pcd = convert_to_point_cloud(ans_collider, rays_collider)
            collider_pcd_sum += collider_pcd

        t2 = time.time()

        # Voxel downsample each separately
        mesh_pcd_subsampled = mesh_pcd_sum.voxel_down_sample(VOXEL_SIZE)
        collider_pcd_subsampled = collider_pcd_sum.voxel_down_sample(VOXEL_SIZE)

        t3 = time.time()

        # Convert to numpy and create node types
        mesh_points = np.asarray(mesh_pcd_subsampled.points)
        collider_points = np.asarray(collider_pcd_subsampled.points)

        # Concatenate points and create node type array
        pcd_combined = np.concatenate([mesh_points, collider_points], axis=0)
        node_types = np.concatenate([
            np.zeros(len(mesh_points), dtype=np.int32),  # 0 for mesh
            np.ones(len(collider_points), dtype=np.int32)  # 1 for collider
        ])

        # FPS sample on combined point cloud
        fps_indices = fpsample.bucket_fps_kdtree_sampling(pcd_combined, num_final_points)
        fps_subsampled = pcd_combined[fps_indices]
        node_types_subsampled = node_types[fps_indices]

    else:
        # No collider - original behavior but with node types
        pcd_sum = o3d.t.geometry.PointCloud().to_legacy()
        for camera_eye, up in zip(camera_eyes, ups):
            ans, rays = do_raycasting(mesh_scene, fov_deg=80, center=center, eye=camera_eye, up=up)
            pcd = convert_to_point_cloud(ans, rays)
            pcd_sum += pcd

        t2 = time.time()
        pcd_subsampled = pcd_sum.voxel_down_sample(VOXEL_SIZE)
        t3 = time.time()
        pcd_subsampled = np.asarray(pcd_subsampled.points)

        # FPS sample
        fps_indices = fpsample.bucket_fps_kdtree_sampling(pcd_subsampled, num_final_points)
        fps_subsampled = pcd_subsampled[fps_indices]
        # All nodes are mesh type (0)
        node_types_subsampled = np.zeros(len(fps_subsampled), dtype=np.int32)

    t4 = time.time()
    if visualize:
        print(f"init: {t1 - t0}")
        print(f"raycasting: {t2 - t1}")
        print(f"voxel downsample: {t3 - t2}")
        print(f"fps: {t4 - t3}")
    if visualize:
        vis_pcd(fps_subsampled, camera_eyes, vertices_tensor.numpy(), node_types_subsampled)
    return fps_subsampled, node_types_subsampled


def do_raycasting(scene, fov_deg: int, center: list, eye: list, up: list) -> tuple:
    """
    Performs the actual raycasting using the provided camera settings
    Args:
        scene: o3d scene object
        center: center of camera
        eye: orientation of camera
        up: up direction of camera
        fov_deg: field of view of camera

    Returns:
        ans: Dictionary of raycasting
        rays: resulting rays of raycasting
    """
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(fov_deg=fov_deg, center=center, eye=eye, up=up,
                                                              width_px=1280 // 2, height_px=720 // 2)
    # We can directly pass the rays tensor to the cast_rays function.
    ans = scene.cast_rays(rays)
    return ans, rays


def convert_to_point_cloud(ans: Dict, rays):
    """
    Converts the hits from raycasting to a point cloud by evaluating the rays at the 'hit' indices
    Args:
        ans: dictionary from raycasting
        rays: rays from raycasting

    Returns:
        pcd: point cloud
    """
    hit = ans['t_hit'].isfinite()
    points = rays[hit][:, :3] + rays[hit][:, 3:] * ans['t_hit'][hit].reshape((-1, 1))
    pcd = o3d.t.geometry.PointCloud(points)
    return pcd.to_legacy()


def convert_to_point_cloud_by_geometry(ans: Dict, rays):
    """
    Converts the hits from raycasting to separate point clouds for mesh and collider
    Args:
        ans: dictionary from raycasting
        rays: rays from raycasting

    Returns:
        mesh_pcd: point cloud for mesh
        collider_pcd: point cloud for collider
    """
    hit = ans['t_hit'].isfinite()
    points = rays[hit][:, :3] + rays[hit][:, 3:] * ans['t_hit'][hit].reshape((-1, 1))

    # Separate by geometry_id
    geometry_ids = ans['geometry_ids'][hit].numpy()
    mesh_points = points[geometry_ids == 0]
    collider_points = points[geometry_ids == 1]

    mesh_pcd = o3d.t.geometry.PointCloud(mesh_points)
    collider_pcd = o3d.t.geometry.PointCloud(collider_points)

    return mesh_pcd.to_legacy(), collider_pcd.to_legacy()


def get_tissue_pcd(pcd, ans: Dict):
    """
    Returns the point cloud points of the tissue from the pointcloud of the scene using their hit indices from raycasting
    Args:
        pcd: point cloud object
        ans: ans dictionary from raycasting
    """
    hit = ans['t_hit'].isfinite()
    geometry_ids = ans['geometry_ids'][hit].numpy()
    indices = np.where(geometry_ids == 0)[0]
    pcd_select = pcd.select_by_index(indices)

    return pcd_select


def get_random_box_position(low: np.array, high: np.array) -> np.array:
    """
    Generate a random position on a box in 3D
    Args:
        low: (left, back, down) point
        high: (right, front, up) point

    Returns:
        direction
    """
    # low = np.array([-0.03, -0.005, 0.1025])
    # high = np.array([0.07, -0.005, 0.1925 ])
    width_x = (high - low)[0]
    width_z = (high - low)[2]
    width_sum = width_x + width_z
    probabilities = np.array([width_z / width_sum, width_x / width_sum])
    on_x_edge = bool(np.random.choice([True, False], p=probabilities))
    if on_x_edge:
        target_x = np.random.choice(np.asarray([low[0], high[0]]))
        target_z = np.random.uniform(low[2], high[2])
    else:
        target_x = np.random.uniform(low[0], high[0])
        target_z = np.random.choice(np.asarray([low[2], high[2]]))
    target_y = high[1]
    target = np.asarray([target_x, target_y, target_z])
    return target


def random_three_vector() -> tuple:
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Returns:
        np.array containing the direction
    """
    phi = np.random.uniform(0, np.pi * 2)
    costheta = np.random.uniform(-1, 1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return (x, y, z)


def random_2D_dir() -> np.array:
    """
    Generates a random 2D unit vector (direction) (x, z) on a circle
    Returns:
        np.array containing the direction
    """
    dir = 2 * math.pi * np.random.uniform(0, 1)
    vx = math.cos(dir)
    vy = 0.0
    vz = math.sin(dir)

    return np.array([vx, vy, vz])


def random_2D_down_dir() -> np.array:
    """
    Generates a random 2D unit vector (direction) (x, z) on the quarter (with center looking down) of a circle
    Returns:
        np.array containing the direction
    """
    dir = 2 * math.pi * np.random.uniform(5 / 8.0, 7 / 8.0)  # before 10 and 14
    vx = math.cos(dir)
    vy = 0.0
    vz = math.sin(dir)

    return np.array([vx, vy, vz])


def custom_draw_geometry(pcd_list: list):
    """
    Custom function to plot a list of point cloud using o3d
    Args:
        pcd_list: list of o3d point objects
    """
    vis = o3d.visualization.Visualizer()

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.mesh_show_wireframe = True
        return False

    vis.create_window()
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.register_animation_callback(change_background_to_black)
    vis.run()
    vis.destroy_window()


def check_for_invalid_rollout(rollout_data_dict, iteration: int) -> bool:
    """
    Args:
        rollout_data_dict: Dict containing all information of the current rollout
        iteration: iteration in which the invalid data point occured
    Returns:
        bool: if the trajectory should be saved (or discarded if invalid)
    """
    save_trajectory = True
    for graph_number, graph in enumerate(rollout_data_dict['tissue_mesh_positions']):
        if not np.isfinite(np.max(graph)):
            print("Infinite position detected")
            print("Traj: ", iteration)
            print("Graph: ", graph_number)
            save_trajectory = False
            break
    return save_trajectory


def get_camera_eyes(dataset_name):
    if dataset_name == "db":
        distance = 5.0

        camera_eyes = [[0.0, 0.0, distance],
                       ]
        ups = [
            [0.0, 1.0, 0.0],
        ]
        center = [0.0, 0.0, 0.0]
        return camera_eyes, ups, center
    elif dataset_name == "sd":
        distance = 5.0

        camera_eyes = [[0.75, 0.75, distance],
                       ]
        ups = [
            [0.0, 1.0, 0.0],
        ]
        center = [0.0, 0.0, 0.0]
        return camera_eyes, ups, center
    elif dataset_name == "trampoline":
        camera_eyes = [
            # [-440, -250, 240],  # left-front diagonal
            [440, -250, 240],  # right-front diagonal
            # [0, 500, 240],  # back
            # [0, 0, -400],  # bottom (looking up)
        ]
        ups = [
            # [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            # [0.0, 0.0, 1.0],
            # [0.0, 1.0, 0.0],  # bottom cam: z is down, so use y as up
        ]
        center = [0.0, 0.0, 100.0]
        return camera_eyes, ups, center
    elif dataset_name == "bbv":
        distance = 20.0

        camera_eyes = [[6.0, 0.0, distance],
                       ]
        ups = [
            [0.0, 1.0, 0.0],
        ]
        center = [6.0, 0.0, 0.0]
        return camera_eyes, ups, center
    else:
        raise ValueError(f"Dataset name `{dataset_name}` not recognized")


def vis_pcd(pcd, camera_eyes, mesh_pos, node_types=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
    import numpy as np

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # --- plot point cloud ---
    if node_types is not None:
        # Color by node type: 0 = mesh (blue), 1 = collider (orange)
        colors = np.where(node_types == 0, 'b', 'orange')
        mesh_mask = node_types == 0
        collider_mask = node_types == 1

        if mesh_mask.any():
            ax.scatter(
                pcd[mesh_mask, 0], pcd[mesh_mask, 1], pcd[mesh_mask, 2],
                c='b', s=3, alpha=0.6, label='Mesh PC'
            )
        if collider_mask.any():
            ax.scatter(
                pcd[collider_mask, 0], pcd[collider_mask, 1], pcd[collider_mask, 2],
                c='orange', s=3, alpha=0.6, label='Collider PC'
            )
    else:
        ax.scatter(
            pcd[:, 0], pcd[:, 1], pcd[:, 2],
            c='b', s=3, alpha=0.6, label='PointCloud'
        )

    # --- plot mesh vertices ---
    if mesh_pos is not None:
        ax.scatter(
            mesh_pos[:, 0], mesh_pos[:, 1], mesh_pos[:, 2],
            c='r', s=6, alpha=0.4, label='Mesh Verts'
        )
    else:
        mesh_pos = np.zeros((0, 3))  # to avoid errors in aspect ratio calculation

    # --- plot cameras ---
    if camera_eyes is not None:
        for cam in camera_eyes:
            ax.scatter(cam[0], cam[1], cam[2], c='g', s=40, marker='^')
    else:
        camera_eyes = np.zeros((0, 3))

    # --- Set equal aspect ratio (important!!) ---
    X = np.concatenate([pcd[:, 0], mesh_pos[:, 0], np.array(camera_eyes)[:, 0]])
    Y = np.concatenate([pcd[:, 1], mesh_pos[:, 1], np.array(camera_eyes)[:, 1]])
    Z = np.concatenate([pcd[:, 2], mesh_pos[:, 2], np.array(camera_eyes)[:, 2]])

    max_range = np.array([
        X.max() - X.min(),
        Y.max() - Y.min(),
        Z.max() - Z.min()
    ]).max() / 2.0

    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("Point Cloud Visualization (3D)")
    ax.legend()

    plt.tight_layout()
    plt.show()
