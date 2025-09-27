# Image Processing Fundamentals

## Overview
Image processing forms the foundation of computer vision, encompassing techniques for enhancing, analyzing, and transforming digital images. This section covers the theoretical foundations, mathematical concepts, and classical algorithms that underpin modern computer vision systems.

## 1. Digital Image Representation

### 1.1 Pixel and Image Basics
A digital image is a 2D array of pixels, where each pixel represents light intensity or color information.

**Mathematical Representation:**
- **Grayscale Image**: $I(x,y) \in [0, 255]$ where $(x,y)$ are spatial coordinates
- **Color Image (RGB)**: $I(x,y) = [R(x,y), G(x,y), B(x,y)]$ where each channel $\in [0, 255]$
- **Binary Image**: $B(x,y) \in \{0, 1\}$

**Image Dimensions:**
- Width × Height (spatial dimensions)
- Number of channels (depth)
- Bit depth (precision per channel)

### 1.2 Color Spaces
Different color representations for various applications:

```python
import cv2
import numpy as np

# Color space conversions
def convert_color_spaces(image):
    """Convert between different color spaces"""
    # RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # RGB to Lab
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # RGB to YUV
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    return hsv, lab, yuv

# Color space properties
color_spaces = {
    'RGB': 'Red, Green, Blue - Additive color model',
    'HSV': 'Hue, Saturation, Value - Intuitive color representation',
    'Lab': 'Lightness, a, b - Perceptually uniform',
    'YUV': 'Luma, Chrominance - Video compression'
}
```

### 1.3 Image Properties and Statistics

```python
def analyze_image_properties(image):
    """Analyze basic image properties"""
    properties = {
        'shape': image.shape,
        'dtype': image.dtype,
        'size': image.size,
        'min_intensity': np.min(image),
        'max_intensity': np.max(image),
        'mean_intensity': np.mean(image),
        'std_intensity': np.std(image)
    }

    # Channel-wise statistics for color images
    if len(image.shape) == 3:
        channels = ['B', 'G', 'R']
        for i, channel in enumerate(channels):
            properties[f'{channel}_mean'] = np.mean(image[:,:,i])
            properties[f'{channel}_std'] = np.std(image[:,:,i])

    return properties
```

## 2. Image Enhancement Techniques

### 2.1 Spatial Domain Operations

#### 2.1.1 Point Operations
Operations applied independently to each pixel:

```python
def point_operations(image):
    """Apply various point operations"""
    # Linear transformations
    brightness_enhanced = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    contrast_enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

    # Gamma correction
    gamma_corrected = np.power(image/255.0, 0.8) * 255
    gamma_corrected = np.uint8(gamma_corrected)

    # Histogram equalization
    if len(image.shape) == 2:  # Grayscale
        hist_eq = cv2.equalizeHist(image)
    else:  # Color image
        # Convert to YUV and equalize Y channel
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        hist_eq = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    return brightness_enhanced, contrast_enhanced, gamma_corrected, hist_eq
```

#### 2.1.2 Histogram Processing

```python
def histogram_analysis(image):
    """Comprehensive histogram analysis"""
    if len(image.shape) == 3:
        # Split channels for color images
        channels = cv2.split(image)
        colors = ['b', 'g', 'r']
        hist_data = {}

        for channel, color in zip(channels, colors):
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            hist_data[color] = hist

        return hist_data
    else:
        # Single channel histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return {'gray': hist}

def adaptive_histogram_equalization(image):
    """Contrast Limited Adaptive Histogram Equalization"""
    if len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)
    else:
        # Apply CLAHE to each channel in Lab color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

### 2.2 Spatial Filtering

#### 2.2.1 Linear Filters (Convolution)

```python
def create_kernels():
    """Create various convolution kernels"""
    kernels = {
        'box_3x3': np.ones((3,3), np.float32) / 9,
        'box_5x5': np.ones((5,5), np.float32) / 25,
        'gaussian_3x3': np.array([[1,2,1],[2,4,2],[1,2,1]], np.float32) / 16,
        'gaussian_5x5': np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],
                                  [4,16,24,16,4],[1,4,6,4,1]], np.float32) / 256,
        'sharpen': np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], np.float32),
        'edge_detection': np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], np.float32),
        'sobel_x': np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32),
        'sobel_y': np.array([[-1,-2,-1],[0,0,0],[1,2,1]], np.float32),
        'laplacian': np.array([[0,1,0],[1,-4,1],[0,1,0]], np.float32)
    }
    return kernels

def apply_filters(image, kernels):
    """Apply convolution filters to image"""
    results = {}

    for name, kernel in kernels.items():
        # Apply filter
        filtered = cv2.filter2D(image, -1, kernel)
        results[name] = filtered

    return results

def gaussian_blur(image, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur with specified parameters"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def median_filter(image, kernel_size=5):
    """Apply median filter for noise reduction"""
    return cv2.medianBlur(image, kernel_size)

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filter for edge-preserving smoothing"""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
```

### 2.3 Frequency Domain Processing

#### 2.3.1 Fourier Transform

```python
def fourier_transform_analysis(image):
    """Analyze image in frequency domain"""
    if len(image.shape) == 3:
        # Convert to grayscale for FFT
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute 2D FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

    return {
        'fourier_transform': f_transform,
        'shifted_spectrum': f_shift,
        'magnitude_spectrum': magnitude_spectrum
    }

def frequency_domain_filtering(image, low_cutoff=30, high_cutoff=100):
    """Apply frequency domain filtering"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)

    # Create filters
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2

    # Low pass filter
    mask_low = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask_low, (ccol, crow), low_cutoff, 1, -1)

    # High pass filter
    mask_high = np.ones((rows, cols), np.uint8)
    cv2.circle(mask_high, (ccol, crow), high_cutoff, 0, -1)

    # Apply filters
    f_low = f_shift * mask_low
    f_high = f_shift * mask_high

    # Inverse FFT
    img_low = np.fft.ifft2(np.fft.ifftshift(f_low)).real
    img_high = np.fft.ifft2(np.fft.ifftshift(f_high)).real

    return np.uint8(img_low), np.uint8(img_high)
```

## 3. Geometric Transformations

### 3.1 Basic Transformations

```python
def geometric_transformations(image):
    """Apply various geometric transformations"""
    h, w = image.shape[:2]

    transformations = {}

    # Translation
    M_translate = np.float32([[1,0,50],[0,1,30]])
    translated = cv2.warpAffine(image, M_translate, (w,h))
    transformations['translation'] = translated

    # Rotation
    M_rotate = cv2.getRotationMatrix2D((w//2, h//2), 45, 1.0)
    rotated = cv2.warpAffine(image, M_rotate, (w,h))
    transformations['rotation'] = rotated

    # Scaling
    scaled_up = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    scaled_down = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    transformations['scaled_up'] = scaled_up
    transformations['scaled_down'] = scaled_down

    # Flipping
    flipped_h = cv2.flip(image, 1)  # Horizontal flip
    flipped_v = cv2.flip(image, 0)  # Vertical flip
    transformations['flip_horizontal'] = flipped_h
    transformations['flip_vertical'] = flipped_v

    return transformations

def perspective_transform(image, src_points, dst_points):
    """Apply perspective transformation"""
    h, w = image.shape[:2]

    # Get perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply perspective transformation
    warped = cv2.warpPerspective(image, M, (w, h))

    return warped, M
```

### 3.2 Image Registration

```python
def feature_based_registration(img1, img2):
    """Register images using feature matching"""
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Match features using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return M, good_matches, kp1, kp2

    return None, [], [], []
```

## 4. Morphological Operations

### 4.1 Basic Morphological Operations

```python
def morphological_operations(image):
    """Apply various morphological operations"""
    # Convert to binary if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Binarize
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Define structuring elements
    kernel_3x3 = np.ones((3,3), np.uint8)
    kernel_5x5 = np.ones((5,5), np.uint8)

    # Basic operations
    erosion = cv2.erode(binary, kernel_3x3, iterations=1)
    dilation = cv2.dilate(binary, kernel_3x3, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_3x3)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_3x3)

    # Advanced operations
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel_3x3)
    tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel_3x3)
    blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel_3x3)

    return {
        'binary': binary,
        'erosion': erosion,
        'dilation': dilation,
        'opening': opening,
        'closing': closing,
        'gradient': gradient,
        'tophat': tophat,
        'blackhat': blackhat
    }
```

### 4.2 Advanced Morphological Operations

```python
def advanced_morphology(image):
    """Apply advanced morphological techniques"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Adaptive thresholding
    binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)

    # Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_adaptive)

    # Skeletonization
    def skeletonize(binary_img):
        """Skeletonize binary image"""
        size = np.size(binary_img)
        skel = np.zeros(binary_img.shape, np.uint8)

        ret, img = cv2.threshold(binary_img, 127, 255, 0)
        img = img // 255
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        return skel * 255

    skeleton = skeletonize(binary_adaptive)

    return {
        'adaptive_binary': binary_adaptive,
        'connected_components': (num_labels, labels, stats, centroids),
        'skeleton': skeleton
    }
```

## 5. Edge Detection and Feature Extraction

### 5.1 Edge Detection Algorithms

```python
def edge_detection_algorithms(image):
    """Apply various edge detection algorithms"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    edges = {}

    # Sobel edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_direction = np.arctan2(sobel_y, sobel_x)

    edges['sobel_x'] = np.uint8(np.absolute(sobel_x))
    edges['sobel_y'] = np.uint8(np.absolute(sobel_y))
    edges['sobel_magnitude'] = np.uint8(sobel_magnitude)

    # Canny edge detection
    edges_canny = cv2.Canny(gray, 50, 150)
    edges['canny'] = edges_canny

    # Laplacian edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edges['laplacian'] = np.uint8(np.absolute(laplacian))

    # Prewitt edge detection
    prewitt_x = cv2.filter2D(gray, -1, np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float32))
    prewitt_y = cv2.filter2D(gray, -1, np.array([[-1,-1,-1],[0,0,0],[1,1,1]], np.float32))
    prewitt_magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)

    edges['prewitt_x'] = np.uint8(np.absolute(prewitt_x))
    edges['prewitt_y'] = np.uint8(np.absolute(prewitt_y))
    edges['prewitt_magnitude'] = np.uint8(prewitt_magnitude)

    return edges
```

### 5.2 Corner and Feature Detection

```python
def corner_detection(image):
    """Detect corners and keypoints"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    corners = {}

    # Harris corner detection
    harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    corners['harris'] = harris_corners

    # Shi-Tomasi corner detection
    shi_tomasi_corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners['shi_tomasi'] = shi_tomasi_corners

    # FAST corner detection
    fast_corners = cv2.FastFeatureDetector_create()
    fast_keypoints = fast_corners.detect(gray, None)
    corners['fast'] = fast_keypoints

    # SIFT keypoints
    sift = cv2.SIFT_create()
    sift_keypoints = sift.detect(gray, None)
    corners['sift'] = sift_keypoints

    # ORB keypoints
    orb = cv2.ORB_create()
    orb_keypoints = orb.detect(gray, None)
    corners['orb'] = orb_keypoints

    return corners
```

### 5.3 Feature Description and Matching

```python
def feature_description_and_matching(img1, img2):
    """Extract features and match between images"""
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1, gray2 = img1, img2

    # Initialize feature detectors
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create()

    # Detect and compute features
    kp1_sift, des1_sift = sift.detectAndCompute(gray1, None)
    kp2_sift, des2_sift = sift.detectAndCompute(gray2, None)

    kp1_orb, des1_orb = orb.detectAndCompute(gray1, None)
    kp2_orb, des2_orb = orb.detectAndCompute(gray2, None)

    # Match SIFT features
    if des1_sift is not None and des2_sift is not None:
        bf_sift = cv2.BFMatcher()
        matches_sift = bf_sift.knnMatch(des1_sift, des2_sift, k=2)

        # Apply ratio test
        good_matches_sift = []
        for m, n in matches_sift:
            if m.distance < 0.75 * n.distance:
                good_matches_sift.append(m)
    else:
        good_matches_sift = []

    # Match ORB features
    if des1_orb is not None and des2_orb is not None:
        bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches_orb = bf_orb.match(des1_orb, des2_orb)

        # Sort matches by distance
        matches_orb = sorted(matches_orb, key=lambda x: x.distance)
    else:
        matches_orb = []

    return {
        'sift': {
            'keypoints_img1': kp1_sift,
            'keypoints_img2': kp2_sift,
            'matches': good_matches_sift
        },
        'orb': {
            'keypoints_img1': kp1_orb,
            'keypoints_img2': kp2_orb,
            'matches': matches_orb
        }
    }
```

## 6. Image Quality Assessment

### 6.1 Quality Metrics

```python
def image_quality_metrics(original, processed):
    """Calculate various image quality metrics"""
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        original_gray, processed_gray = original, processed

    metrics = {}

    # Mean Squared Error
    mse = np.mean((original_gray - processed_gray) ** 2)
    metrics['mse'] = mse

    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    metrics['psnr'] = psnr

    # Structural Similarity Index
    ssim = cv2.SSIM(original_gray, processed_gray)
    metrics['ssim'] = ssim

    # Universal Quality Index
    def universal_quality_index(img1, img2):
        """Calculate Universal Quality Index"""
        mean1 = np.mean(img1)
        mean2 = np.mean(img2)
        var1 = np.var(img1)
        var2 = np.var(img2)
        cov = np.cov(img1.flatten(), img2.flatten())[0,1]

        numerator = 4 * mean1 * mean2 * cov
        denominator = (mean1**2 + mean2**2) * (var1 + var2)

        if denominator == 0:
            return 0
        else:
            return numerator / denominator

    uqi = universal_quality_index(original_gray, processed_gray)
    metrics['uqi'] = uqi

    return metrics
```

### 6.2 Noise Analysis

```python
def noise_analysis(image):
    """Analyze noise characteristics in image"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    noise_analysis_results = {}

    # Estimate noise using Laplacian
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise_analysis_results['laplacian_variance'] = laplacian_var

    # Estimate noise standard deviation
    def estimate_noise_std(image_patch):
        """Estimate noise standard deviation"""
        if len(image_patch.shape) == 3:
            patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
        else:
            patch = image_patch

        # Use high-pass filter to extract noise
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        noise = cv2.filter2D(patch.astype(float), -1, kernel)
        return np.std(noise)

    noise_std = estimate_noise_std(gray)
    noise_analysis_results['noise_std'] = noise_std

    # Signal-to-Noise Ratio
    signal_power = np.mean(gray**2)
    noise_power = noise_std**2
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    noise_analysis_results['snr'] = snr

    return noise_analysis_results
```

## 7. Practical Applications and Exercises

### 7.1 Image Enhancement Pipeline

```python
def complete_image_enhancement_pipeline(image):
    """Complete image enhancement pipeline"""
    enhanced_images = {}

    # Step 1: Noise reduction
    denoised = bilateral_filter(image, d=9, sigma_color=75, sigma_space=75)
    enhanced_images['denoised'] = denoised

    # Step 2: Contrast enhancement
    if len(image.shape) == 3:
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(denoised)

    enhanced_images['enhanced'] = enhanced

    # Step 3: Sharpening
    sharpening_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], np.float32)
    sharpened = cv2.filter2D(enhanced, -1, sharpening_kernel)
    enhanced_images['sharpened'] = sharpened

    return enhanced_images
```

### 7.2 Feature-based Image Alignment

```python
def feature_based_image_alignment(template, target):
    """Align target image to template using features"""
    # Convert to grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(template_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            # Warp template image
            h, w = template.shape[:2]
            aligned = cv2.warpPerspective(template, M, (w, h))

            return aligned, M, good_matches, kp1, kp2

    return None, None, [], [], []
```

## 8. Mathematical Foundations

### 8.1 Convolution Theorem

The convolution theorem states that convolution in the spatial domain is equivalent to multiplication in the frequency domain:

$$f * g = \mathcal{F}^{-1}\{\mathcal{F}\{f\} \cdot \mathcal{F}\{g\}\}$$

Where:
- $f * g$ is the convolution of functions $f$ and $g$
- $\mathcal{F}$ denotes the Fourier transform
- $\mathcal{F}^{-1}$ denotes the inverse Fourier transform

### 8.2 Image Statistics

**Mean:**
$$\mu = \frac{1}{MN}\sum_{i=0}^{M-1}\sum_{j=0}^{N-1}I(i,j)$$

**Variance:**
$$\sigma^2 = \frac{1}{MN}\sum_{i=0}^{M-1}\sum_{j=0}^{N-1}(I(i,j) - \mu)^2$$

**Standard Deviation:**
$$\sigma = \sqrt{\sigma^2}$$

### 8.3 Gaussian Filter

2D Gaussian filter kernel:
$$G(x,y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}$$

Where $\sigma$ is the standard deviation controlling the amount of smoothing.

### 8.4 Sobel Operators

**Horizontal Sobel:**
$$S_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$

**Vertical Sobel:**
$$S_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

**Gradient Magnitude:**
$$|\nabla I| = \sqrt{S_x^2 + S_y^2}$$

**Gradient Direction:**
$$\theta = \arctan\left(\frac{S_y}{S_x}\right)$$

This comprehensive foundation provides the mathematical and algorithmic basis for advanced computer vision techniques that build upon these fundamental image processing concepts.