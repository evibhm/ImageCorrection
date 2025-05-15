import cv2
import gradio as gr
import numpy as np
from scipy import signal


def resizeImg(image, height=900):
    """
    根据指定高度等比例缩放图像。
    Resizes an image proportionally to a specified height.
    """
    h, w = image.shape[:2]
    if h == 0:  # 防止除以零 Prevent division by zero
        return image
    pro = height / h
    size = (int(w * pro), int(height))
    img = cv2.resize(image, size)
    return img


def getCanny(image, thresh1, thresh2, dilate_iterations):
    """
    对图像进行边缘检测。
    Performs Canny edge detection on an image.
    Input 'image' is expected to be BGR for this function's context.
    Parameters:
    - image: Input BGR image
    - thresh1: First threshold for the hysteresis procedure.
    - thresh2: Second threshold for the hysteresis procedure.
    - dilate_iterations: Number of iterations for dilation.
    """
    # 如果是彩色图像，先转为灰度图 Convert to grayscale if it's a color image
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3 and image.shape[2] == 3
        else image
    )
    # 高斯滤波 Gaussian blur
    binary = cv2.GaussianBlur(src=gray, ksize=(3, 3), sigmaX=2.0, sigmaY=2.0)
    # 边缘检测 Edge detection
    binary = cv2.Canny(binary, thresh1, thresh2, apertureSize=3)
    # 膨胀操作，尽量使边缘闭合 Dilate to close gaps in edges
    kernel = np.ones((3, 3), np.uint8)
    if dilate_iterations > 0:  # Apply dilation only if iterations > 0
        binary = cv2.dilate(binary, kernel, iterations=dilate_iterations)
    return binary  # Returns a grayscale image


def findMaxContour(image):
    """
    找出图像中面积最大的轮廓。
    Finds the contour with the largest area in an image.
    Input 'image' is expected to be a binary image.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = 0.0
    max_contour = []
    for contour in contours:
        currentArea = cv2.contourArea(contour)
        if currentArea > max_area:
            max_area = currentArea
            max_contour = contour
    return max_contour, max_area


def getBoxPoint(contour):
    """
    通过多边形拟合凸包获取四个顶点。
    Gets four corner points by fitting a polygon to the convex hull of a contour.
    """
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(
        contour, True
    )  # Epsilon factor could also be a parameter
    approx = cv2.approxPolyDP(hull, epsilon, True)
    approx = approx.reshape((len(approx), 2))
    return approx


def adaPoint(box, pro):
    """
    根据比例调整点集坐标。
    Adapts point set coordinates based on a scaling factor.
    'pro' = scaled_dimension / original_dimension. Scales 'box' by 1/pro.
    """
    box_pro = box.astype(np.float32)
    if pro != 0:
        box_pro = box_pro / pro
    return box_pro.astype(np.float32)


def orderPoints(pts):
    """
    对四边形的四个顶点进行排序：[top-left, top-right, bottom-right, bottom-left]。
    Orders the four vertices of a quadrilateral.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def pointDistance(a, b):
    """
    计算两点之间的欧氏距离。
    Calculates the Euclidean distance between two points.
    """
    return int(np.sqrt(np.sum(np.square(a - b))))


def warpImage(image, box):
    """
    对图像进行透视变换。
    Performs perspective warping on an image. Input 'image' is BGR.
    """
    box = np.array(box, dtype="float32")
    if box.shape[0] != 4:
        print("警告: 透视变换需要4个角点。将返回原始图像的副本。")
        return image.copy()

    ordered_box = orderPoints(box)

    widthA = pointDistance(ordered_box[0], ordered_box[1])
    widthB = pointDistance(ordered_box[3], ordered_box[2])
    maxWidth = max(int(widthA), int(widthB))

    heightA = pointDistance(ordered_box[0], ordered_box[3])
    heightB = pointDistance(ordered_box[1], ordered_box[2])
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth <= 0 or maxHeight <= 0:  # Ensure dimensions are positive
        print("警告: 计算出的变换后图像宽度或高度无效。将返回原始图像的副本。")
        return image.copy()

    dst_rect = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    try:
        M = cv2.getPerspectiveTransform(ordered_box, dst_rect)
        warped = cv2.warpPerspective(
            image, M, (maxWidth, maxHeight)
        )  # Warped image is BGR
    except cv2.error as e:
        print(f"透视变换时发生OpenCV错误: {e}")
        return image.copy()  # Return copy of original on error
    return warped  # Returns BGR image


def adaptive_thres(img, win, beta):
    """
    对图像进行自适应阈值处理。
    Applies adaptive thresholding. Input 'img' can be BGR or Grayscale.
    Parameters:
    - img: Input image (BGR or Grayscale)
    - win: Window size for local mean calculation (should be odd).
    - beta: Coefficient to adjust threshold based on local mean.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        img_gray = img
    else:
        print("错误：adaptive_thres 的输入图像格式不正确。")
        return img

    # Ensure window size is odd
    if win % 2 == 0:
        win = win + 1 if win > 0 else 3  # Make it odd, ensure positive

    kern = np.ones([win, win])
    sums = signal.correlate2d(img_gray, kern, "same", boundary="symm")
    cnts = signal.correlate2d(np.ones_like(img_gray), kern, "same", boundary="symm")

    cnts[cnts == 0] = 1
    means = sums / cnts

    binary_img = np.where(img_gray < means * beta, 0, 255)
    return binary_img.astype(np.uint8)  # Returns grayscale binary image


def enhance(
    image_bgr,
    canny_thresh1,
    canny_thresh2,
    dilate_iter,
    adaptive_win,
    adaptive_beta_val,
):
    """
    图像增强主函数。
    Returns: (final_output, canny_output, contour_img_output, warped_color_output)
    """
    if image_bgr is None:
        print("错误：输入到 enhance 函数的图像为空。")
        return None, None, None, None

    # Default final_output to a grayscale version of the original image_bgr
    # This ensures that if processing fails early, final_output is still a valid image.
    final_output = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    canny_intermediate = None
    contour_img_intermediate = None
    warped_color_intermediate = None

    original_height, original_width = image_bgr.shape[:2]
    if original_height == 0 or original_width == 0:
        print("警告：输入图像尺寸无效。")
        # Returns default final_output (grayscale of original), others None
        return final_output, None, None, None

    target_processing_height = 900
    img_for_canny = resizeImg(image_bgr.copy(), height=target_processing_height)

    # 1. Canny Edges using parameters
    canny_intermediate = getCanny(
        img_for_canny, canny_thresh1, canny_thresh2, dilate_iter
    )

    if original_height == 0:
        scaling_ratio_for_adapoint = 1.0
    else:
        scaling_ratio_for_adapoint = target_processing_height / original_height

    max_contour, _ = findMaxContour(canny_intermediate)

    if not len(max_contour):
        print("警告：未找到主要轮廓。")
        # Returns default final_output, canny_intermediate, others None
        return final_output, canny_intermediate, None, None

    boxes_on_resized = getBoxPoint(max_contour)
    if boxes_on_resized.shape[0] < 4:
        print("警告: 轮廓点不足4个，无法形成四边形。")
        return final_output, canny_intermediate, None, None

    boxes_on_original_scale = adaPoint(boxes_on_resized, scaling_ratio_for_adapoint)
    ordered_boxes_on_original = orderPoints(boxes_on_original_scale.astype(np.float32))

    contour_img_intermediate = image_bgr.copy()
    cv2.drawContours(
        contour_img_intermediate,
        [ordered_boxes_on_original.astype(np.int32)],
        -1,
        (0, 255, 0),
        3,  # Green contour
    )

    warped_color_intermediate = warpImage(image_bgr, ordered_boxes_on_original)

    if (
        warped_color_intermediate is None
        or warped_color_intermediate.size < 100  # Check for minimal size
        or np.array_equal(warped_color_intermediate, image_bgr.copy())
    ):  # Check if warp returned original
        print("警告：透视变换失败或结果无效。")
        if np.array_equal(warped_color_intermediate, image_bgr.copy()):
            warped_color_intermediate = (
                None  # Explicitly set to None if it's just a copy
            )
        # Return current state, final_output is still default (grayscale of original)
        return (
            final_output,
            canny_intermediate,
            contour_img_intermediate,
            warped_color_intermediate,
        )

    # 7. Final Result using parameters, only if warped_color_intermediate is valid
    final_output = adaptive_thres(
        warped_color_intermediate, adaptive_win, adaptive_beta_val
    )

    return (
        final_output,
        canny_intermediate,
        contour_img_intermediate,
        warped_color_intermediate,
    )


# --- Gradio 接口函数 ---
def process_image_gradio(
    input_image_rgb,
    canny_t1,
    canny_t2,
    dilate_iterations,
    adaptive_window,
    adaptive_beta_slider,
):
    """
    Gradio 接口的处理函数。
    Returns: (canny_gray, contour_rgb, warped_rgb, final_gray)
    """
    if input_image_rgb is None:  # If no image is uploaded or image is cleared
        # Return None for all 4 outputs to clear them
        return None, None, None, None

    input_image_bgr = cv2.cvtColor(input_image_rgb, cv2.COLOR_RGB2BGR)

    # Call the enhance function with all parameters
    # enhance returns: (final_gray, canny_gray, contour_bgr, warped_bgr)
    final_result_gray, canny_result_gray, contour_result_bgr, warped_result_bgr = (
        enhance(
            input_image_bgr,
            int(canny_t1),
            int(canny_t2),
            int(dilate_iterations),
            int(adaptive_window),
            float(adaptive_beta_slider),
        )
    )

    # Convert BGR results to RGB for Gradio display
    contour_result_rgb = None
    if contour_result_bgr is not None:
        contour_result_rgb = cv2.cvtColor(contour_result_bgr, cv2.COLOR_BGR2RGB)

    warped_result_rgb = None
    if warped_result_bgr is not None:
        if (
            len(warped_result_bgr.shape) == 3 and warped_result_bgr.shape[2] == 3
        ):  # Check if it's a color image
            warped_result_rgb = cv2.cvtColor(warped_result_bgr, cv2.COLOR_BGR2RGB)
        else:  # It might be grayscale if warpImage returned an error image that was grayscale
            warped_result_rgb = warped_result_bgr  # Pass as is if already grayscale

    # Handle cases where enhance might return None for final_result_gray (e.g., early exit)
    # Ensure a valid grayscale image is returned for the final output slot if processing fails partially
    if final_result_gray is None and input_image_rgb is not None:
        final_result_gray = cv2.cvtColor(input_image_rgb, cv2.COLOR_RGB2GRAY)

    # Return the four processed images
    return (canny_result_gray, contour_result_rgb, warped_result_rgb, final_result_gray)


# 定义 Gradio 界面
with gr.Blocks(theme=gr.Theme()) as demo:
    gr.Markdown("# 图片边缘检测、矫正与增强锐化系统")
    gr.Markdown("上传图片并调整参数以查看不同处理阶段的效果。")

    with gr.Row():
        # Input Controls Column
        with gr.Column(scale=1, min_width=350):
            input_image_rgb = gr.Image(
                type="numpy",
                label="上传图片 (Upload Image)",
                sources=["upload", "webcam", "clipboard"],
            )

            gr.Markdown("#### Canny 边缘检测参数")
            canny_t1 = gr.Slider(
                minimum=1, maximum=200, value=60, step=1, label="低阈值 (Thresh1)"
            )
            canny_t2 = gr.Slider(
                minimum=50, maximum=400, value=240, step=1, label="高阈值 (Thresh2)"
            )
            dilate_iterations = gr.Slider(
                minimum=0,
                maximum=10,
                value=1,
                step=1,
                label="膨胀次数 (Dilation Iter.)",
            )

            gr.Markdown("#### 自适应阈值参数")
            adaptive_window = gr.Slider(
                minimum=3,
                maximum=51,
                value=9,
                step=2,
                label="窗口大小 (Win Size - odd)",
            )
            adaptive_beta_slider = gr.Slider(
                minimum=0.8,
                maximum=1.2,
                value=0.9,
                step=0.01,
                label="Beta 系数 (Beta Coeff.)",
            )

        # Output Images Column
        with gr.Column(scale=3):
            gr.Markdown("### 处理结果预览 (Processing Results Preview)")
            # 2x2 grid for the four output images
            with gr.Row():
                canny_out = gr.Image(
                    type="numpy", label="1. Canny 边缘", interactive=False, height=250
                )
                contour_out = gr.Image(
                    type="numpy", label="2. 轮廓检测", interactive=False, height=250
                )
            with gr.Row():
                warped_out = gr.Image(
                    type="numpy", label="3. 透视矫正图", interactive=False, height=250
                )
                final_out = gr.Image(
                    type="numpy", label="4. 自适应二值", interactive=False, height=250
                )

    # Consolidate inputs and outputs lists for event handling
    # inputs_list remains the same
    inputs_list = [
        input_image_rgb,
        canny_t1,
        canny_t2,
        dilate_iterations,
        adaptive_window,
        adaptive_beta_slider,
    ]
    # outputs_list now has 4 components
    outputs_list = [canny_out, contour_out, warped_out, final_out]

    # Event handling for live updates
    for comp in inputs_list:
        comp.change(
            fn=process_image_gradio,
            inputs=inputs_list,
            outputs=outputs_list,
            show_progress="minimal",
        )

    # Example loading
    gr.Examples(
        examples=[
            ["examples/document_sample.png", 60, 240, 1, 9, 0.9],
            ["examples/receipt_sample.jpg", 60, 240, 1, 9, 0.9],
            ["examples/example3.jpg", 25, 80, 1, 9, 0.9],
            ["examples/example4.jpg", 60, 240, 1, 9, 0.97],
            ["examples/example5.jpg", 30, 50, 5, 9, 0.9],
            ["examples/example6.jpg", 30, 50, 1, 9, 0.96],
        ],  # Example data structure remains the same for inputs
        inputs=inputs_list,  # inputs for the function call
        outputs=outputs_list,  # components to update with the function's return values
        fn=process_image_gradio,
        label="示例图片 (Click to load an example)",
    )

# --- 启动 Gradio 应用 ---
if __name__ == "__main__":
    demo.launch()
