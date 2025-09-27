# 3D Computer Vision and Geometry

## Overview
3D computer vision deals with understanding and reconstructing three-dimensional scenes from 2D images and sensors. This section covers the mathematical foundations, geometric principles, and algorithms for 3D perception, reconstruction, and analysis.

## 1. Camera Geometry and Projection

### 1.1 Camera Model

#### 1.1.1 Pinhole Camera Model
The pinhole camera model describes how 3D points project onto a 2D image plane:

**Forward Projection:**
$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z_c} \mathbf{K} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
$$

Where:
- $(u, v)$: Image coordinates
- $(X_c, Y_c, Z_c)$: Camera coordinates
- $\mathbf{K}$: Camera intrinsic matrix

**Camera Intrinsic Matrix:**
$$
\mathbf{K} = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

- $f_x, f_y$: Focal lengths in pixels
- $(c_x, c_y)$: Principal point
- $s$: Skew parameter

#### 1.1.2 Camera Extrinsic Parameters
Transform from world to camera coordinates:

$$
\mathbf{X}_c = \mathbf{R} \mathbf{X}_w + \mathbf{t}
$$

Where:
- $\mathbf{R}$: 3×3 rotation matrix
- $\mathbf{t}$: 3×1 translation vector

**Complete Projection:**
$$
\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} | \mathbf{t}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

### 1.2 Lens Distortion

#### 1.2.1 Radial Distortion
$$
\begin{aligned}
x_{\text{corrected}} &= x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) \\
y_{\text{corrected}} &= y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
\end{aligned}
$$

#### 1.2.2 Tangential Distortion
$$
\begin{aligned}
x_{\text{corrected}} &= x + 2p_1 xy + p_2(r^2 + 2x^2) \\
y_{\text{corrected}} &= y + p_1(r^2 + 2y^2) + 2p_2 xy
\end{aligned}
$$

Where $r^2 = x^2 + y^2$ and $(x, y)$ are normalized image coordinates.

```python
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraModel:
    """Camera model with intrinsic and extrinsic parameters"""
    def __init__(self, fx, fy, cx, cy, skew=0, distortion_coeffs=None):
        self.K = np.array([[fx, skew, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float64)

        self.distortion_coeffs = distortion_coeffs if distortion_coeffs is not None else np.zeros(5)

    def project_points(self, points_3d, R=None, t=None):
        """Project 3D points to 2D image coordinates"""
        if R is None:
            R = np.eye(3)
        if t is None:
            t = np.zeros(3)

        # Transform to camera coordinates
        points_camera = R @ points_3d.T + t.reshape(-1, 1)
        points_camera = points_camera.T

        # Project to image plane
        points_homogeneous = self.K @ points_camera.T
        points_2d = points_homogeneous[:2, :] / points_homogeneous[2, :]

        return points_2d.T

    def undistort_points(self, points_2d):
        """Remove lens distortion from 2D points"""
        return cv2.undistortPoints(points_2d.reshape(-1, 1, 2), self.K, self.distortion_coeffs)

    def distort_points(self, points_2d):
        """Apply lens distortion to 2D points"""
        # Convert to normalized coordinates
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        x = (points_2d[:, 0] - cx) / fx
        y = (points_2d[:, 1] - cy) / fy

        # Apply distortion
        k1, k2, p1, p2, k3 = self.distortion_coeffs
        r2 = x**2 + y**2

        # Radial distortion
        x_distorted = x * (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)
        y_distorted = y * (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)

        # Tangential distortion
        x_distorted += 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        y_distorted += p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

        # Convert back to pixel coordinates
        u_distorted = x_distorted * fx + cx
        v_distorted = y_distorted * fy + cy

        return np.column_stack([u_distorted, v_distorted])
```

## 2. Epipolar Geometry

### 2.1 Essential and Fundamental Matrices

#### 2.1.1 Essential Matrix
Relates corresponding points in two camera views:

$$
\mathbf{x}_2^T \mathbf{E} \mathbf{x}_1 = 0
$$

Where $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$ and $[\mathbf{t}]_\times$ is the skew-symmetric matrix of translation vector $\mathbf{t}$.

**Skew-symmetric matrix:**
$$
[\mathbf{t}]_\times = \begin{bmatrix} 0 & -t_z & t_y \\ t_z & 0 & -t_x \\ -t_y & t_x & 0 \end{bmatrix}
$$

#### 2.1.2 Fundamental Matrix
Relates corresponding points in two images with different intrinsics:

$$
\mathbf{x}_2^T \mathbf{F} \mathbf{x}_1 = 0
$$

Where $\mathbf{F} = \mathbf{K}_2^{-T} \mathbf{E} \mathbf{K}_1^{-1}$

```python
class EpipolarGeometry:
    """Epipolar geometry operations"""
    def __init__(self, camera1, camera2):
        self.camera1 = camera1
        self.camera2 = camera2

    def compute_fundamental_matrix(self, points1, points2):
        """Compute fundamental matrix from point correspondences"""
        if len(points1) >= 8:
            F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
            return F, mask
        return None, None

    def compute_essential_matrix(self, points1, points2):
        """Compute essential matrix from point correspondences"""
        # Normalize points
        points1_norm = self.camera1.undistort_points(points1).squeeze()
        points2_norm = self.camera2.undistort_points(points2).squeeze()

        # Compute essential matrix
        E, mask = cv2.findEssentialMat(points1_norm, points2_norm,
                                      method=cv2.RANSAC, prob=0.999, threshold=1.0)
        return E, mask

    def recover_pose(self, E, points1, points2):
        """Recover camera pose from essential matrix"""
        points1_norm = self.camera1.undistort_points(points1).squeeze()
        points2_norm = self.camera2.undistort_points(points2).squeeze()

        _, R, t, mask = cv2.recoverPose(E, points1_norm, points2_norm,
                                      self.camera1.K)
        return R, t, mask

    def compute_epipolar_lines(self, points, F, which_image=2):
        """Compute epipolar lines"""
        if which_image == 2:
            lines = cv2.computeCorrespondEpilines(points.reshape(-1, 1, 2), 1, F)
        else:
            lines = cv2.computeCorrespondEpilines(points.reshape(-1, 1, 2), 2, F)
        return lines

    def triangulate_points(self, points1, points2, R, t):
        """Triangulate 3D points from two views"""
        # Projection matrices
        P1 = self.camera1.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.camera2.K @ np.hstack([R, t])

        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        points_3d = points_4d[:3, :] / points_4d[3, :]

        return points_3d.T
```

## 3. Structure from Motion (SfM)

### 3.1 Feature Detection and Matching

```python
class SFMReconstruction:
    """Structure from Motion pipeline"""
    def __init__(self, camera_model):
        self.camera = camera_model
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

    def detect_and_match_features(self, img1, img2):
        """Detect and match features between two images"""
        # Detect keypoints and compute descriptors
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        # Match features
        matches = self.matcher.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Extract matched points
        points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        return points1, points2, good_matches, kp1, kp2

    def incremental_reconstruction(self, images):
        """Incremental structure from motion"""
        reconstructions = []

        # Start with first pair
        points1, points2, matches, kp1, kp2 = self.detect_and_match_features(
            images[0], images[1]
        )

        # Compute essential matrix
        epipolar_geom = EpipolarGeometry(self.camera, self.camera)
        E, mask = epipolar_geom.compute_essential_matrix(points1, points2)

        # Recover pose
        R, t, pose_mask = epipolar_geom.recover_pose(E, points1, points2)

        # Triangulate initial points
        points_3d = epipolar_geom.triangulate_points(
            points1[mask.ravel() == 1], points2[mask.ravel() == 1], R, t
        )

        # Initialize reconstruction
        reconstruction = {
            'cameras': [
                {'R': np.eye(3), 't': np.zeros(3)},
                {'R': R, 't': t}
            ],
            'points_3d': points_3d,
            'point_indices': np.where(mask.ravel() == 1)[0]
        }

        return reconstruction

    def bundle_adjustment(self, reconstruction, image_points):
        """Bundle adjustment optimization"""
        # This would typically use Ceres Solver or g2o
        # Simplified implementation using PyTorch
        def bundle_adjustment_loss(cameras, points_3d, observations):
            total_loss = 0
            for i, (cam_idx, point_idx, observed) in enumerate(observations):
                # Project 3D point
                R = cameras[cam_idx]['R']
                t = cameras[cam_idx]['t']

                projected = self.camera.project_points(points_3d[point_idx:point_idx+1], R, t)
                loss = F.mse_loss(torch.tensor(projected), torch.tensor(observed))
                total_loss += loss

            return total_loss

        return bundle_adjustment_loss
```

## 4. Multi-View Stereo

### 4.1 Stereo Vision

```python
class StereoVision:
    """Stereo vision for depth estimation"""
    def __init__(self, camera_left, camera_right, baseline):
        self.camera_left = camera_left
        self.camera_right = camera_right
        self.baseline = baseline

    def compute_disparity(self, img_left, img_right):
        """Compute disparity map using stereo matching"""
        # Convert to grayscale
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Stereo matching
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=7,
            P1=8*3*7**2,
            P2=32*3*7**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        disparity = stereo.compute(gray_left, gray_right)
        return disparity

    def disparity_to_depth(self, disparity):
        """Convert disparity to depth"""
        fx = self.camera_left.K[0, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = fx * self.baseline / disparity
            depth[disparity <= 0] = 0
        return depth

    def compute_point_cloud(self, img_left, disparity):
        """Generate 3D point cloud from disparity"""
        depth = self.disparity_to_depth(disparity)

        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Back-project to 3D
        points_3d = np.zeros((h, w, 3))
        points_3d[:,:,0] = (u - self.camera_left.K[0, 2]) * depth / self.camera_left.K[0, 0]
        points_3d[:,:,1] = (v - self.camera_left.K[1, 2]) * depth / self.camera_left.K[1, 1]
        points_3d[:,:,2] = depth

        # Remove invalid points
        valid = disparity > 0
        points_3d = points_3d[valid]

        return points_3d
```

### 4.2 Multi-View Stereo (MVS)

```python
class MultiViewStereo:
    """Multi-view stereo reconstruction"""
    def __init__(self, cameras):
        self.cameras = cameras

    def patch_match_stereo(self, images, depth_range, num_planes):
        """PatchMatch stereo algorithm"""
        height, width = images[0].shape[:2]

        # Initialize depth planes
        depth_planes = np.linspace(depth_range[0], depth_range[1], num_planes)

        # Random initialization
        depth_map = np.random.uniform(depth_range[0], depth_range[1], (height, width))

        # PatchMatch iterations
        for iteration in range(5):
            # Propagation
            depth_map = self.propagate_depth(depth_map, images)

            # Random refinement
            depth_map = self.refine_depth(depth_map, images, depth_range)

        return depth_map

    def propagate_depth(self, depth_map, images):
        """Spatial propagation of depth values"""
        height, width = depth_map.shape

        # Create copy for updated values
        new_depth_map = depth_map.copy()

        # Horizontal propagation
        for y in range(height):
            for x in range(1, width):
                # Propagate from left
                if self.evaluate_energy(images, x, y, depth_map[y, x-1]) < \
                   self.evaluate_energy(images, x, y, depth_map[y, x]):
                    new_depth_map[y, x] = depth_map[y, x-1]

        # Vertical propagation
        for y in range(1, height):
            for x in range(width):
                # Propagate from top
                if self.evaluate_energy(images, x, y, depth_map[y-1, x]) < \
                   self.evaluate_energy(images, x, y, depth_map[y, x]):
                    new_depth_map[y, x] = depth_map[y-1, x]

        return new_depth_map

    def evaluate_energy(self, images, x, y, depth):
        """Evaluate photo-consistency energy"""
        # Compute photo-consistency across views
        total_cost = 0
        reference_patch = self.extract_patch(images[0], x, y)

        for i in range(1, len(images)):
            # Project point to other views
            warped_patch = self.warp_patch(images[i], x, y, depth, i)
            cost = self.compute_ncc(reference_patch, warped_patch)
            total_cost += cost

        return total_cost / (len(images) - 1)

    def extract_patch(self, image, x, y, patch_size=5):
        """Extract image patch"""
        half_size = patch_size // 2
        return image[max(0, y-half_size):min(image.shape[0], y+half_size+1),
                     max(0, x-half_size):min(image.shape[1], x+half_size+1)]
```

## 5. SLAM (Simultaneous Localization and Mapping)

### 5.1 Visual SLAM

```python
class VisualSLAM:
    """Visual SLAM system"""
    def __init__(self, camera_model):
        self.camera = camera_model
        self.feature_detector = cv2.ORB_create()
        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        # Map and pose
        self.map_points = []
        self.keyframes = []
        self.current_pose = np.eye(4)
        self.frame_count = 0

    def process_frame(self, image):
        """Process a single frame"""
        # Detect features
        keypoints, descriptors = self.feature_detector.detectAndCompute(image, None)

        if self.frame_count == 0:
            # Initialize map
            self.initialize_map(image, keypoints, descriptors)
        else:
            # Track frame
            self.track_frame(image, keypoints, descriptors)

        self.frame_count += 1

    def initialize_map(self, image, keypoints, descriptors):
        """Initialize map from first frame"""
        self.last_keypoints = keypoints
        self.last_descriptors = descriptors
        self.last_image = image

        # Add first keyframe
        self.keyframes.append({
            'pose': self.current_pose.copy(),
            'keypoints': keypoints,
            'descriptors': descriptors,
            'image': image.copy()
        })

    def track_frame(self, image, keypoints, descriptors):
        """Track current frame against last keyframe"""
        # Match features
        matches = self.descriptor_matcher.match(
            self.last_descriptors, descriptors
        )

        if len(matches) < 10:
            # Lost tracking, relocalize or initialize new map
            return False

        # Extract matched points
        points1 = np.float32([self.last_keypoints[m.queryIdx].pt for m in matches])
        points2 = np.float32([keypoints[m.trainIdx].pt for m in matches])

        # Estimate motion (simplified PnP)
        if len(self.map_points) > 0:
            # PnP with known 3D points
            success, rvec, tvec = cv2.solvePnP(
                self.map_points, points1, self.camera.K, self.camera.distortion_coeffs
            )

            if success:
                # Update pose
                R, _ = cv2.Rodrigues(rvec)
                self.current_pose[:3, :3] = R
                self.current_pose[:3, 3] = tvec.flatten()

        return True

    def triangulate_new_points(self, matches):
        """Triangulate new map points"""
        if len(self.keyframes) < 2:
            return

        # Get last two keyframes
        kf1 = self.keyframes[-2]
        kf2 = self.keyframes[-1]

        # Extract corresponding points
        points1 = np.float32([kf1['keypoints'][m.queryIdx].pt for m in matches])
        points2 = np.float32([kf2['keypoints'][m.trainIdx].pt for m in matches])

        # Triangulate
        epipolar_geom = EpipolarGeometry(self.camera, self.camera)
        R = kf2['pose'][:3, :3] @ kf1['pose'][:3, :3].T
        t = kf2['pose'][:3, 3] - R @ kf1['pose'][:3, 3]

        points_3d = epipolar_geom.triangulate_points(points1, points2, R, t)

        # Add to map
        for i, point_3d in enumerate(points_3d):
            self.map_points.append(point_3d)

    def local_mapping(self):
        """Local mapping and bundle adjustment"""
        if len(self.keyframes) < 3:
            return

        # Recent keyframes
        recent_keyframes = self.keyframes[-3:]

        # Optimize recent map (simplified)
        self.optimize_local_map(recent_keyframes)

    def optimize_local_map(self, keyframes):
        """Optimize local map using bundle adjustment"""
        # Simplified bundle adjustment
        # In practice, would use g2o or Ceres Solver
        pass

    def loop_closure(self):
        """Detect and close loops"""
        if len(self.keyframes) < 10:
            return

        # Check for loop closure with earlier keyframes
        current_kf = self.keyframes[-1]

        for i, kf in enumerate(self.keyframes[:-10]):
            # Match features
            matches = self.descriptor_matcher.match(
                kf['descriptors'], current_kf['descriptors']
            )

            if len(matches) > 50:  # Loop detected
                self.close_loop(i, len(self.keyframes) - 1)
                break

    def close_loop(self, kf_idx1, kf_idx2):
        """Close loop between two keyframes"""
        # Compute relative pose
        # In practice, would use pose graph optimization
        pass
```

## 6. Neural Radiance Fields (NeRF)

### 6.1 Neural Radiance Fields Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Positional encoding for NeRF"""
    def __init__(self, input_dim, max_freq_log2, num_freqs):
        super(PositionalEncoding, self).__init__()

        self.input_dim = input_dim
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs

        # Create frequency bands
        self.freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs)

    def forward(self, x):
        """Apply positional encoding"""
        # x: (..., input_dim)
        shape = x.shape

        # Reshape for broadcasting
        x = x.reshape(-1, self.input_dim)  # (N, input_dim)

        # Apply positional encoding
        encoded = [x]
        for freq in self.freq_bands:
            encoded.extend([torch.sin(freq * x), torch.cos(freq * x)])

        # Concatenate all encodings
        encoded = torch.cat(encoded, dim=-1)

        return encoded.reshape(*shape[:-1], -1)

class NeRF(nn.Module):
    """Neural Radiance Fields model"""
    def __init__(self, input_dim=3, output_dim=4, hidden_dim=256,
                 num_layers=8, max_freq_log2=10, num_freqs=10):
        super(NeRF, self).__init__()

        # Positional encoding
        self.pos_encoder = PositionalEncoding(input_dim, max_freq_log2, num_freqs)

        # Direction encoding (view-dependent)
        self.dir_encoder = PositionalEncoding(3, max_freq_log2=4, num_freqs=4)

        # Input dimension after encoding
        encoded_dim = input_dim + 2 * input_dim * num_freqs

        # Network layers
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(encoded_dim, hidden_dim))

        # Hidden layers with skip connection
        for i in range(num_layers - 1):
            if i == num_layers // 2:
                # Skip connection
                self.layers.append(nn.Linear(hidden_dim + encoded_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layers
        self.sigma_layer = nn.Linear(hidden_dim, 1)
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        self.rgb_layer = nn.Linear(hidden_dim + self.dir_encoder.output_dim, 3)

        self.activation = nn.ReLU()

    def forward(self, positions, directions):
        """Forward pass"""
        # Encode positions
        pos_encoded = self.pos_encoder(positions)

        # Encode directions
        dir_encoded = self.dir_encoder(directions)

        # Forward through network
        x = pos_encoded

        for i, layer in enumerate(self.layers):
            if i == len(self.layers) // 2:
                # Skip connection
                x = torch.cat([x, pos_encoded], dim=-1)
            x = self.activation(layer(x))

        # Predict density
        sigma = self.sigma_layer(x)

        # Extract features
        features = self.feature_layer(x)

        # Predict RGB (view-dependent)
        rgb_input = torch.cat([features, dir_encoded], dim=-1)
        rgb = torch.sigmoid(self.rgb_layer(rgb_input))

        return torch.cat([rgb, sigma], dim=-1)

class NeRFRenderer:
    """NeRF rendering engine"""
    def __init__(self, nerf_model, near=0.1, far=10.0, num_samples=64):
        self.nerf = nerf_model
        self.near = near
        self.far = far
        self.num_samples = num_samples

    def render_rays(self, ray_origins, ray_directions):
        """Render rays through the scene"""
        batch_size = ray_origins.shape[0]

        # Sample points along rays
        t_vals = torch.linspace(self.near, self.far, self.num_samples)
        t_vals = t_vals.expand(batch_size, -1)

        # Add random perturbation
        if self.training:
            t_vals = t_vals + torch.rand_like(t_vals) * (self.far - self.near) / self.num_samples

        # Compute 3D points
        points = ray_origins.unsqueeze(1) + t_vals.unsqueeze(2) * ray_directions.unsqueeze(1)

        # Flatten for network input
        points_flat = points.reshape(-1, 3)
        directions_flat = ray_directions.unsqueeze(1).expand(-1, self.num_samples, -1)
        directions_flat = directions_flat.reshape(-1, 3)

        # Query network
        rgb_sigma = self.nerf(points_flat, directions_flat)

        # Reshape
        rgb = rgb_sigma[..., :3].reshape(batch_size, self.num_samples, 3)
        sigma = rgb_sigma[..., 3:].reshape(batch_size, self.num_samples)

        # Compute weights using volume rendering
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).expand(batch_size, 1)], dim=-1)

        alpha = 1 - torch.exp(-sigma * dists)
        weights = alpha * torch.cumprod(1 - alpha + 1e-10, dim=-1)

        # Compute color
        rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)

        # Compute depth
        depth_map = torch.sum(weights * t_vals, dim=1)

        return rgb_map, depth_map, weights

class GaussianSplatting:
    """3D Gaussian Splatting representation"""
    def __init__(self, num_points=10000):
        self.num_points = num_points

        # Initialize Gaussian parameters
        self.positions = nn.Parameter(torch.randn(num_points, 3))
        self.scales = nn.Parameter(torch.ones(num_points, 3) * 0.1)
        self.rotations = nn.Parameter(torch.randn(num_points, 4))  # Quaternions
        self.colors = nn.Parameter(torch.rand(num_points, 3))
        self.opacities = nn.Parameter(torch.ones(num_points) * 0.5)

    def render(self, camera_pos, camera_rot, image_size):
        """Render scene using Gaussian splatting"""
        # Transform Gaussians to camera space
        # Implement Gaussian splatting rendering
        # This is a simplified version
        pass
```

## 7. Practical Applications

### 7.1 3D Reconstruction Pipeline

```python
class ReconstructionPipeline:
    """Complete 3D reconstruction pipeline"""
    def __init__(self):
        self.camera_calibrator = None
        self.sfm_reconstructor = None
        self.mvs_reconstructor = None

    def calibrate_camera(self, calibration_images, pattern_size=(9, 6)):
        """Camera calibration using checkerboard"""
        # Prepare object points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        camera = CameraModel(mtx[0,0], mtx[1,1], mtx[0,2], mtx[1,2],
                           distortion_coeffs=dist)

        return camera, ret

    def reconstruct_from_images(self, images):
        """Complete reconstruction from multiple images"""
        # 1. Camera calibration
        camera, _ = self.calibrate_camera(images)

        # 2. Structure from Motion
        sfm = SFMReconstruction(camera)
        reconstruction = sfm.incremental_reconstruction(images)

        # 3. Multi-View Stereo
        mvs = MultiViewStereo([camera] * len(images))
        depth_maps = []

        for i in range(len(images)):
            if i > 0:
                depth_map = mvs.patch_match_stereo(
                    [images[0], images[i]], (0.1, 10.0), 64
                )
                depth_maps.append(depth_map)

        return {
            'camera': camera,
            'reconstruction': reconstruction,
            'depth_maps': depth_maps
        }
```

This comprehensive foundation covers the mathematical principles and practical implementations of 3D computer vision, from basic camera geometry to advanced neural representations like NeRF, providing the essential knowledge for understanding and implementing 3D vision systems.