import warnings
import logging
import os
import re
import torch
import numpy as np
from PIL import Image
import json
from huggingface_hub import snapshot_download
import sys
from pathlib import Path
import glob
import cv2
import math
import matplotlib.colors
import tempfile
from torchvision import transforms

# --- 抑制警告 ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")

# --- 日志配置 ---
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("deepspeed").setLevel(logging.ERROR)

# --- 环境检测 ---
IS_COMFYUI_ENV = False
try:
    import comfy
    IS_COMFYUI_ENV = True
except ImportError:
    pass

if IS_COMFYUI_ENV:
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)

# --- Mock 必要模块（非ComfyUI环境兼容）---
try:
    import folder_paths
except ImportError:
    class MockFolderPaths:
        models_dir = os.path.expanduser("~/models")
        supported_pt_extensions = {".pt", ".pth", ".ckpt", ".safetensors"}
        folder_names_and_paths = {}
        
        @staticmethod
        def add_model_folder_path(name, path, is_default=False):
            if name not in MockFolderPaths.folder_names_and_paths:
                MockFolderPaths.folder_names_and_paths[name] = []
            if path not in MockFolderPaths.folder_names_and_paths[name]:
                MockFolderPaths.folder_names_and_paths[name].append(path)
        
        @staticmethod
        def get_filename_list(folder_name):
            if folder_name not in MockFolderPaths.folder_names_and_paths:
                return []
            file_list = []
            for folder in MockFolderPaths.folder_names_and_paths[folder_name]:
                if os.path.exists(folder):
                    for f in os.listdir(folder):
                        file_list.append(f)
            return list(set(file_list))
        
        @staticmethod
        def get_full_path(folder_name, filename):
            if folder_name not in MockFolderPaths.folder_names_and_paths:
                return None
            for folder in MockFolderPaths.folder_names_and_paths[folder_name]:
                full_path = os.path.join(folder, filename)
                if os.path.exists(full_path):
                    return full_path
            return None
        
        @staticmethod
        def get_folder_paths(folder_name):
            return MockFolderPaths.folder_names_and_paths.get(folder_name, [])
        
        @staticmethod
        def get_output_directory():
            output_dir = os.path.expanduser("~/outputs")
            os.makedirs(output_dir, exist_ok=True)
            return output_dir
        
        @staticmethod
        def get_save_image_path(filename_prefix, output_dir, width, height):
            base_filename = filename_prefix
            full_output_folder = output_dir
            os.makedirs(full_output_folder, exist_ok=True)
            return full_output_folder, base_filename, "", width, height
    
    folder_paths = MockFolderPaths()

try:
    import model_management
except ImportError:
    class MockModelManagement:
        @staticmethod
        def get_torch_device():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        @staticmethod
        def soft_empty_cache():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    model_management = MockModelManagement()

# --- 核心依赖导入 ---
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel

package_path = str(Path(__file__).parent)
if package_path not in sys.path:
    sys.path.insert(0, package_path)

try:
    from .models.HeatmapHead import get_heatmap_head
    from .models.ModifiedUNet import Modified_forward
    from .pipelines.SDPose_D_Pipeline import SDPose_D_Pipeline
except ImportError as e:
    print("="*50)
    print("SDPose Node: 关键错误 - 缺失核心模块")
    print("请确保 'models'、'pipelines'、'mmpose' 子文件夹完整")
    print(f"原始错误: {e}")
    print("="*50)
    raise e

from safetensors.torch import load_file

# --- YOLO导入 ---
YOLO_AVAILABLE = False
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("SDPose Node: 未找到 ultralytics，YOLO检测功能禁用")
except Exception as e:
    print(f"SDPose Node: YOLO导入失败: {e}，功能禁用")

# --- 修复：变量名大小写统一 ---
SDPOSE_MODEL_DIR = os.path.join(folder_paths.models_dir, "SDPose_OOD")
YOLO_MODEL_DIR = os.path.join(folder_paths.models_dir, "yolo")
GROUNDINGDINO_MODEL_DIR = os.path.join(folder_paths.models_dir, "grounding-dino")

# 注册模型文件夹（使用正确的全大写变量名）
folder_paths.add_model_folder_path("SDPose_OOD", SDPOSE_MODEL_DIR, is_default=True)
folder_paths.add_model_folder_path("yolo", YOLO_MODEL_DIR, is_default=False)
folder_paths.add_model_folder_path("grounding-dino", GROUNDINGDINO_MODEL_DIR, is_default=False)

# 创建目录
os.makedirs(SDPOSE_MODEL_DIR, exist_ok=True)
os.makedirs(YOLO_MODEL_DIR, exist_ok=True)
os.makedirs(GROUNDINGDINO_MODEL_DIR, exist_ok=True)

# --- 空嵌入文件路径 ---
NODE_DIR = Path(__file__).parent
EMPTY_EMBED_DIR = os.path.join(NODE_DIR, "empty_text_encoder")
os.makedirs(EMPTY_EMBED_DIR, exist_ok=True)

# --- 张量/图像转换工具（修复：强化类型转换和容错）---
def tensor_to_pil(tensor):
    if tensor.dim() != 3:
        raise ValueError(f"期望3D张量，实际得到{tensor.dim()}D")
    # 强制裁剪到0-1范围，避免超出取值范围
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return Image.fromarray((tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

def pil_to_tensor(image):
    if not isinstance(image, Image.Image):
        raise TypeError(f"期望PIL图像，实际得到{type(image)}")
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def numpy_to_tensor(img_np):
    # 强制转换为uint8，避免数据类型异常
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)

# --- OpenPose标准校验辅助函数 ---
def validate_openpose_black_image(pose_image):
    """校验纯黑背景姿态图是否符合OpenPose标准"""
    # 1. 格式校验：3通道、uint8、纯黑背景（大部分像素值为0）
    if pose_image.ndim != 3 or pose_image.shape[2] != 3 or pose_image.dtype != np.uint8:
        return False, "格式错误：需为3通道uint8数组"
    # 2. 背景校验：纯黑像素占比>99%（OpenPose姿态图仅骨架有非黑像素）
    black_pixel_ratio = (pose_image == 0).all(axis=2).sum() / (pose_image.shape[0] * pose_image.shape[1])
    if black_pixel_ratio < 0.99:
        return False, f"背景非纯黑：黑像素占比{black_pixel_ratio:.4f} < 0.99"
    # 3. 骨架像素校验：存在非黑像素（骨架关键点/连线）
    non_black_pixels = (pose_image != 0).any(axis=2).sum()
    if non_black_pixels == 0:
        return False, "无骨架信息：未检测到非黑像素"
    return True, f"校验通过：黑像素占比{black_pixel_ratio:.4f}，骨架像素数{non_black_pixels}"

# --- 姿态绘图函数（优化：符合OpenPose官方标准，纯黑背景、固定参数、标准色彩）---
def draw_body17_keypoints_openpose_style(canvas, keypoints, scores=None, threshold=0.3, overlay_mode=False, overlay_alpha=0.6, scale_for_xinsr=False):
    # 修复1：画布合法性校验，强制对齐OpenPose标准（3通道uint8纯黑）
    if not isinstance(canvas, np.ndarray) or len(canvas.shape) != 3 or canvas.shape[2] != 3:
        # 强制创建OpenPose标准纯黑画布（默认512x512，与官方一致）
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    # 强制转换为uint8，避免数据类型异常（OpenPose标准数据类型）
    canvas = canvas.astype(np.uint8)
    H, W, C = canvas.shape
    if len(keypoints) < 7:
        return canvas
    
    neck = (keypoints[5] + keypoints[6]) / 2
    neck_score = min(scores[5], scores[6]) if scores is not None and len(scores)>=7 else 1.0
    
    candidate = np.zeros((18, 2))
    candidate_scores = np.zeros(18)
    candidate[0] = keypoints[0]; candidate[1] = neck; candidate[2] = keypoints[6]; candidate[3] = keypoints[8]; candidate[4] = keypoints[10]
    candidate[5] = keypoints[5]; candidate[6] = keypoints[7]; candidate[7] = keypoints[9]; candidate[8] = keypoints[12]; candidate[9] = keypoints[14]
    candidate[10] = keypoints[16]; candidate[11] = keypoints[11]; candidate[12] = keypoints[13]; candidate[13] = keypoints[15]; candidate[14] = keypoints[2]
    candidate[15] = keypoints[1]; candidate[16] = keypoints[4]; candidate[17] = keypoints[3]
    
    if scores is not None and len(scores) >= 17:
        candidate_scores[0] = scores[0]; candidate_scores[1] = neck_score; candidate_scores[2] = scores[6]; candidate_scores[3] = scores[8]; candidate_scores[4] = scores[10]
        candidate_scores[5] = scores[5]; candidate_scores[6] = scores[7]; candidate_scores[7] = scores[9]; candidate_scores[8] = scores[12]; candidate_scores[9] = scores[14]
        candidate_scores[10] = scores[16]; candidate_scores[11] = scores[11]; candidate_scores[12] = scores[13]; candidate_scores[13] = scores[15]; candidate_scores[14] = scores[2]
        candidate_scores[15] = scores[1]; candidate_scores[16] = scores[4]; candidate_scores[17] = scores[3]
    
    # 修复2：采用OpenPose官方固定绘制参数（不动态缩放，确保格式统一）
    # 官方OpenPose标准：骨架宽度=2，关键点半径=3
    stickwidth = 2  # 固定骨架宽度，与OpenPose官方一致
    circle_radius = 3  # 固定关键点半径，与OpenPose官方一致
    
    # 修复3：采用OpenPose官方标准色彩序列（RGB格式，与官方完全对齐）
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], 
               [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
    # OpenPose官方标准色彩（RGB），确保骨架颜色与官方一致
    openpose_colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
        [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
        [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
    ]
    
    # 修复4：绘制时增加坐标合法性校验+OpenPose像素值标准化
    for i in range(len(limbSeq)):
        index = np.array(limbSeq[i]) - 1
        if index[0] >= len(candidate) or index[1] >= len(candidate):
            continue
        if scores is not None and (candidate_scores[index[0]] < threshold or candidate_scores[index[1]] < threshold):
            continue
        
        Y = candidate[index.astype(int), 0]
        X = candidate[index.astype(int), 1]
        # 裁剪坐标到画布范围内（OpenPose不允许越界像素）
        Y = np.clip(Y, 0, W-1)
        X = np.clip(X, 0, H-1)
        
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        
        if length < 1:
            continue
        
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        # 确保多边形坐标在画布内，强制转换为int32（cv2要求格式）
        polygon = np.clip(polygon, 0, np.array([W-1, H-1])).astype(np.int32)
        # 强制使用OpenPose标准色彩，避免色彩偏差
        cv2.fillConvexPoly(canvas, polygon, openpose_colors[i % len(openpose_colors)])
    
    for i in range(18):
        if scores is not None and candidate_scores[i] < threshold:
            continue
        x, y = int(candidate[i][0]), int(candidate[i][1])
        if 0 <= x < W and 0 <= y < H:
            # 绘制关键点，与OpenPose格式一致（实心圆，无描边）
            cv2.circle(canvas, (x, y), circle_radius, openpose_colors[i % len(openpose_colors)], thickness=-1)
    
    # 修复5：最终画布标准化（强制uint8，纯黑背景无杂色，符合OpenPose输入要求）
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return canvas

def draw_wholebody_keypoints_openpose_style(canvas, keypoints, scores=None, threshold=0.3, overlay_mode=False, overlay_alpha=0.6, scale_for_xinsr=False):
    # 修复1：画布合法性校验，强制对齐OpenPose标准（3通道uint8纯黑）
    if not isinstance(canvas, np.ndarray) or len(canvas.shape) != 3 or canvas.shape[2] != 3:
        # 强制创建OpenPose标准纯黑画布（默认512x512，与官方一致）
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    # 强制转换为uint8，避免数据类型异常（OpenPose标准数据类型）
    canvas = canvas.astype(np.uint8)
    H, W, C = canvas.shape
    
    # 采用OpenPose官方固定绘制参数
    stickwidth = 2  # 固定骨架宽度，与OpenPose官方一致
    circle_radius = 3  # 固定关键点半径，与OpenPose官方一致
    
    # OpenPose官方标准色彩序列（RGB）
    body_colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
        [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
        [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
    ]
    
    body_limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], 
                    [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
    hand_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], 
                  [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], 
                  [17, 18], [18, 19], [19, 20]]
    
    if len(keypoints) >= 18:
        for i, limb in enumerate(body_limbSeq):
            idx1, idx2 = limb[0] - 1, limb[1] - 1
            if idx1 >= 18 or idx2 >= 18:
                continue
            if scores is not None and len(scores)>=18 and (scores[idx1] < threshold or scores[idx2] < threshold):
                continue
            
            Y = np.array([keypoints[idx1][0], keypoints[idx2][0]])
            X = np.array([keypoints[idx1][1], keypoints[idx2][1]])
            # 裁剪坐标到画布范围内
            Y = np.clip(Y, 0, W-1)
            X = np.clip(X, 0, H-1)
            
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            
            if length < 1:
                continue
            
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            polygon = np.clip(polygon, 0, np.array([W-1, H-1])).astype(np.int32)
            cv2.fillConvexPoly(canvas, polygon, body_colors[i % len(body_colors)])
        
        for i in range(18):
            if scores is not None and len(scores)>=18 and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), circle_radius, body_colors[i % len(body_colors)], thickness=-1)
    
    if len(keypoints) >= 24:
        for i in range(18, 24):
            if scores is not None and len(scores)>=24 and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), circle_radius, body_colors[i % len(body_colors)], thickness=-1)
    
    if len(keypoints) >= 113:
        for ie, edge in enumerate(hand_edges):
            idx1, idx2 = 92 + edge[0], 92 + edge[1]
            if scores is not None and len(scores)>=113 and (scores[idx1] < threshold or scores[idx2] < threshold):
                continue
            x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
            x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
            
            if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                color = matplotlib.colors.hsv_to_rgb([ie / float(len(hand_edges)), 1.0, 1.0]) * 255
                cv2.line(canvas, (x1, y1), (x2, y2), color.astype(np.uint8).tolist(), thickness=stickwidth)
        
        for i in range(92, 113):
            if scores is not None and len(scores)>=113 and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), circle_radius, (0, 0, 255), thickness=-1)
    
    if len(keypoints) >= 134:
        for ie, edge in enumerate(hand_edges):
            idx1, idx2 = 113 + edge[0], 113 + edge[1]
            if scores is not None and len(scores)>=134 and (scores[idx1] < threshold or scores[idx2] < threshold):
                continue
            x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
            x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
            
            if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                color = matplotlib.colors.hsv_to_rgb([ie / float(len(hand_edges)), 1.0, 1.0]) * 255
                cv2.line(canvas, (x1, y1), (x2, y2), color.astype(np.uint8).tolist(), thickness=stickwidth)
        
        for i in range(113, 134):
            if scores is not None and len(scores)>=134 and i < len(scores) and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), circle_radius, (0, 0, 255), thickness=-1)
    
    if len(keypoints) >= 92:
        for i in range(24, 92):
            if scores is not None and len(scores)>=92 and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), circle_radius - 1, (255, 255, 255), thickness=-1)
    
    # 最终画布标准化，符合OpenPose标准
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return canvas

# --- GroundingDINO相关函数 ---
def load_dino_image(image_pil):
    try:
        from groundingdino.datasets import transforms as T
    except ImportError:
        raise ImportError("请安装 groundingdino-py: pip install groundingdino-py")
    
    if not isinstance(image_pil, Image.Image):
        raise TypeError(f"期望PIL图像，实际得到{type(image_pil)}")
    
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    image, _ = transform(image_pil, None)
    return image

def groundingdino_predict(dino_model_wrapper, image_pil, prompt, threshold):
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)
    
    def get_grounding_output(model, image, caption, box_threshold, device):
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        
        model.to(device)
        image = image.to(device)
        boxes_filt = torch.tensor([])
        
        try:
            with torch.no_grad():
                outputs = model(image[None], captions=[caption])
            
            logits = outputs["pred_logits"].sigmoid()[0]
            boxes = outputs["pred_boxes"][0]
            
            filt_mask = logits.max(dim=1)[0] > box_threshold
            logits_filt = logits[filt_mask]
            boxes_filt = boxes[filt_mask]
            boxes_filt = box_cxcywh_to_xyxy(boxes_filt)
        except Exception as e:
            print(f"SDPose Node: GroundingDINO推理失败: {e}")
        finally:
            model.to("cpu")
            model_management.soft_empty_cache()
        
        return boxes_filt.cpu()
    
    if not isinstance(dino_model_wrapper, dict) or "model" not in dino_model_wrapper:
        print("SDPose Node: 无效的GroundingDINO模型")
        return []
    if not isinstance(prompt, str) or not prompt.strip():
        print("SDPose Node: GroundingDINO提示词为空")
        return []
    if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
        print(f"SDPose Node: 无效的GroundingDINO阈值 {threshold}")
        return []
    
    try:
        dino_image = load_dino_image(image_pil.convert("RGB"))
        device = model_management.get_torch_device()
        boxes_filt_norm = get_grounding_output(dino_model_wrapper["model"], dino_image, prompt, threshold, device)
        
        if boxes_filt_norm.shape[0] == 0:
            return []
        
        H, W = image_pil.size[1], image_pil.size[0]
        boxes_filt_abs = boxes_filt_norm * torch.Tensor([W, H, W, H])
        
        return boxes_filt_abs.tolist()
    except Exception as e:
        print(f"SDPose Node: GroundingDINO预测失败: {e}")
        return []

# --- 核心处理函数（修复：全黑关键问题 - 图像混合、画布、归一化）---
def detect_person_yolo(image, yolo_model_path, confidence_threshold=0.5):
    if not YOLO_AVAILABLE:
        return [[0, 0, image.shape[1], image.shape[0]]], False
    
    if not isinstance(image, np.ndarray) or len(image.shape) != 3:
        return [[0, 0, 512, 512]], False
    if not isinstance(confidence_threshold, (int, float)) or not (0 <= confidence_threshold <= 1):
        confidence_threshold = 0.5
    
    try:
        model = YOLO(yolo_model_path)
        results = model(image, verbose=False)
        person_bboxes = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) == 0 and float(box.conf[0]) > confidence_threshold:
                        person_bboxes.append(box.xyxy[0].cpu().numpy().tolist())
        
        if person_bboxes:
            return person_bboxes, True
        else:
            return [[0, 0, image.shape[1], image.shape[0]]], False
    except Exception as e:
        print(f"SDPose Node: YOLO检测失败: {e}")
        return [[0, 0, image.shape[1] if image.shape else 512, image.shape[0] if image.shape else 512]], False

def preprocess_image_for_sdpose(image, bbox=None, input_size=(768, 1024)):
    if not isinstance(input_size, (list, tuple)) or len(input_size) != 2:
        input_size = (768, 1024)
    
    if isinstance(image, np.ndarray):
        try:
            # 修复：强制转换为RGB，避免BGR转RGB失败
            if image.shape[2] == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image).convert("RGB")
        except Exception as e:
            print(f"SDPose Node: 图像转换失败: {e}")
            pil_image = Image.new("RGB", input_size, color=(255, 255, 255))
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.new("RGB", input_size, color=(255, 255, 255))
    
    original_size = pil_image.size
    crop_info = (0, 0, pil_image.width, pil_image.height)
    
    if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x1, y1, x2, y2 = map(int, bbox)
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= pil_image.width and y2 <= pil_image.height:
                pil_image = pil_image.crop((x1, y1, x2, y2))
                crop_info = (x1, y1, x2 - x1, y2 - y1)
        except Exception:
            pass
    
    transform = transforms.Compose([
        transforms.Resize((input_size[1], input_size[0]), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    input_tensor = transform(pil_image).unsqueeze(0)
    return input_tensor, original_size, crop_info

def restore_keypoints_to_original(keypoints, crop_info, input_size, original_size):
    if not isinstance(keypoints, np.ndarray):
        return np.array([])
    if not isinstance(crop_info, (list, tuple)) or len(crop_info) != 4:
        return keypoints
    if not isinstance(input_size, (list, tuple)) or len(input_size) != 2:
        input_size = (768, 1024)
    if not isinstance(original_size, (list, tuple)) or len(original_size) != 2:
        original_size = (input_size[0], input_size[1])
    
    x1, y1, crop_w, crop_h = crop_info
    scale_x = crop_w / input_size[0] if input_size[0] > 0 else 1.0
    scale_y = crop_h / input_size[1] if input_size[1] > 0 else 1.0
    
    keypoints_restored = keypoints.copy()
    keypoints_restored[:, 0] = keypoints[:, 0] * scale_x + x1
    keypoints_restored[:, 1] = keypoints[:, 1] * scale_y + y1
    
    return keypoints_restored

def convert_to_loader_json(all_keypoints, all_scores, image_width, image_height, keypoint_scheme="body", threshold=0.3):
    if not isinstance(all_keypoints, (list, np.ndarray)):
        all_keypoints = []
    if not isinstance(all_scores, (list, np.ndarray)):
        all_scores = []
    if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
        threshold = 0.3
    image_width = int(image_width) if isinstance(image_width, (int, float)) else 512
    image_height = int(image_height) if isinstance(image_height, (int, float)) else 512
    
    people = []
    zip_data = zip(all_keypoints, all_scores) if len(all_keypoints) == len(all_scores) else []
    
    for keypoints, scores in zip_data:
        person_data = {}
        pose_kpts_18 = []
        
        if keypoint_scheme == "body":
            if len(keypoints) < 17 or len(scores) < 17:
                continue
            
            neck = (keypoints[5] + keypoints[6]) / 2
            neck_score = min(scores[5], scores[6])
            
            op_keypoints = np.zeros((18, 2))
            op_scores = np.zeros(18)
            coco_to_op_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            
            for i_op in range(18):
                if i_op == 1:
                    op_keypoints[i_op] = neck
                    op_scores[i_op] = neck_score
                else:
                    i_coco = coco_to_op_map[i_op]
                    if 0 <= i_coco < len(keypoints):
                        op_keypoints[i_op] = keypoints[i_coco]
                        op_scores[i_op] = scores[i_coco]
            
            for i in range(18):
                score = float(op_scores[i])
                if score < threshold:
                    pose_kpts_18.extend([0.0, 0.0, 0.0])
                else:
                    pose_kpts_18.extend([float(op_keypoints[i, 0]), float(op_keypoints[i, 1]), score])
        else:
            if len(keypoints) < 18 or len(scores) < 18:
                continue
            
            for i in range(18):
                score = float(scores[i]) if i < len(scores) else 0.0
                if score < threshold:
                    pose_kpts_18.extend([0.0, 0.0, 0.0])
                else:
                    x = float(keypoints[i, 0]) if i < len(keypoints) else 0.0
                    y = float(keypoints[i, 1]) if i < len(keypoints) else 0.0
                    pose_kpts_18.extend([x, y, score])
        
        person_data["pose_keypoints_2d"] = pose_kpts_18
        people.append(person_data)
    
    return {"width": image_width, "height": image_height, "people": people}

def convert_to_openpose_json(all_keypoints, all_scores, image_width, image_height, keypoint_scheme="body"):
    if not isinstance(all_keypoints, (list, np.ndarray)):
        all_keypoints = []
    if not isinstance(all_scores, (list, np.ndarray)):
        all_scores = []
    image_width = int(image_width) if isinstance(image_width, (int, float)) else 512
    image_height = int(image_height) if isinstance(image_height, (int, float)) else 512
    
    if image_width == 0 or image_height == 0:
        print("SDPose Node: 警告 - 图像宽高为0，使用默认512x512")
        image_width, image_height = 512, 512
    
    people = []
    zip_data = zip(all_keypoints, all_scores) if len(all_keypoints) == len(all_scores) else []
    
    for person_idx, (keypoints, scores) in enumerate(zip_data):
        person_data = {}
        
        if keypoint_scheme == "body":
            if len(keypoints) < 17 or len(scores) < 17:
                continue
            
            neck = (keypoints[5] + keypoints[6]) / 2
            neck_score = min(scores[5], scores[6]) if scores[5] > 0.3 and scores[6] > 0.3 else 0
            
            op_keypoints = np.zeros((18, 2))
            op_scores = np.zeros(18)
            coco_to_op_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            
            for i_op in range(18):
                if i_op == 1:
                    op_keypoints[i_op] = neck
                    op_scores[i_op] = neck_score
                else:
                    i_coco = coco_to_op_map[i_op]
                    if 0 <= i_coco < len(keypoints):
                        op_keypoints[i_op] = keypoints[i_coco]
                        op_scores[i_op] = scores[i_coco]
            
            pose_kpts_18 = []
            for i in range(18):
                x, y = op_keypoints[i, 0], op_keypoints[i, 1]
                s = float(op_scores[i])
                
                if s < 0.1:
                    pose_kpts_18.extend([0.0, 0.0, 0.0])
                else:
                    pose_kpts_18.extend([
                        float(x) / float(image_width),
                        float(y) / float(image_height),
                        1.0
                    ])
            
            person_data.update({
                "pose_keypoints_2d": pose_kpts_18,
                "foot_keypoints_2d": [],
                "face_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "hand_left_keypoints_2d": []
            })
        else:
            def get_fixed_kpt(i):
                if i >= len(keypoints) or i >= len(scores):
                    return [0.0, 0.0, 0.0]
                
                x, y = keypoints[i, 0], keypoints[i, 1]
                s = float(scores[i]) if i < len(scores) else 0.0
                
                if s < 0.1:
                    return [0.0, 0.0, 0.0]
                
                return [
                    float(x) / float(image_width),
                    float(y) / float(image_height),
                    1.0
                ]
            
            pose_kpts = [v for i in range(18) for v in get_fixed_kpt(i)]
            foot_kpts = [v for i in range(18, 24) for v in get_fixed_kpt(i)]
            face_kpts = [v for i in range(24, 92) for v in get_fixed_kpt(i)]
            right_hand_kpts = [v for i in range(92, 113) for v in get_fixed_kpt(i)]
            left_hand_kpts = [v for i in range(113, 134) for v in get_fixed_kpt(i)]
            
            person_data.update({
                "pose_keypoints_2d": pose_kpts,
                "foot_keypoints_2d": foot_kpts,
                "face_keypoints_2d": face_kpts,
                "hand_right_keypoints_2d": right_hand_kpts,
                "hand_left_keypoints_2d": left_hand_kpts
            })
        
        people.append(person_data)
    
    return {"people": people, "canvas_width": image_width, "canvas_height": image_height}

def _combine_frame_jsons(frame_jsons):
    combined_data = {
        "frame_count": len(frame_jsons),
        "frames": []
    }
    
    for i, frame_json_str in enumerate(frame_jsons):
        try:
            frame_data = {}
            if frame_json_str is None:
                raise ValueError("JSON数据为空")
            if isinstance(frame_json_str, (dict,)):
                frame_data = frame_json_str
            else:
                if isinstance(frame_json_str, (str,)):
                    frame_data = json.loads(frame_json_str)
                else:
                    raise ValueError("不是合法的JSON字符串或字典")
            
            frame_data["frame_index"] = i
            combined_data["frames"].append(frame_data)
        except Exception as e:
            print(f"SDPose Node: 警告 - 帧{i}JSON解析失败: {e}")
            combined_data["frames"].append({"frame_index": i, "error": f"JSON解析失败: {str(e)}"})
    
    return json.dumps(combined_data, indent=2)

# --- 模型加载节点 ---
class YOLOModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("yolo"),),
            }
        }

    RETURN_TYPES = ("YOLO_MODEL",)
    FUNCTION = "load_yolo"
    CATEGORY = "SDPose"

    def load_yolo(self, model_name):
        if not YOLO_AVAILABLE:
            raise ImportError("请安装 ultralytics 以使用YOLO模型")
        
        model_path = folder_paths.get_full_path("yolo", model_name)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到YOLO模型: {model_name}")
        
        print(f"SDPose Node: 加载YOLO模型: {model_path}")
        model = YOLO(model_path)
        return (model,)

class SDPoseOODLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["Body", "WholeBody"],),
                "unet_precision": (["fp32", "fp16", "bf16"],),
                "device": (["auto", "cuda", "cpu"],),
                "unload_on_finish": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SDPOSE_MODEL",)
    FUNCTION = "load_sdpose_model"
    CATEGORY = "SDPose"

    def get_model_path(self, repo_name):
        model_pathes = folder_paths.get_folder_paths("SDPose_OOD")
        for path in model_pathes:
            model_path = os.path.join(path, repo_name)
            if os.path.exists(os.path.join(model_path, "unet")):
                return model_path
        return os.path.join(SDPOSE_MODEL_DIR, repo_name)

    def load_sdpose_model(self, model_type, unet_precision, device, unload_on_finish):
        repo_id = {
            "Body": "teemosliang/SDPose-Body",
            "WholeBody": "teemosliang/SDPose-Wholebody"
        }[model_type]
        
        keypoint_scheme = model_type.lower()
        model_path = self.get_model_path(repo_id.split('/')[-1])
        
        if not os.path.exists(os.path.join(model_path, "unet")):
            print(f"SDPose Node: 下载模型: {repo_id} 到 {model_path}")
            snapshot_download(repo_id=repo_id, local_dir=model_path, local_dir_use_symlinks=False)

        if device == "auto":
            device = model_management.get_torch_device()
        else:
            device = torch.device(device)

        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[unet_precision]
        if device.type == 'cpu' and dtype == torch.float16:
            print("SDPose Node: 警告 - CPU不支持FP16， fallback到FP32")
            dtype = torch.float32

        print(f"SDPose Node: 加载模型到设备: {device}，精度: {unet_precision}")
        
        embed_path = os.path.join(EMPTY_EMBED_DIR, "empty_embedding.safetensors")
        if not os.path.exists(embed_path):
            raise FileNotFoundError(f"未找到空嵌入文件: {embed_path}，请运行 generate_empty_embedding.py")
        empty_text_embed = load_file(embed_path)["empty_text_embed"].to(device)

        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype).to(device)
        unet = Modified_forward(unet, keypoint_scheme=keypoint_scheme)
        vae = AutoencoderKL.from_pretrained(model_path, subfolder='vae').to(device)
        
        dec_path = os.path.join(model_path, "decoder", "decoder.safetensors")
        hm_decoder = get_heatmap_head(mode=keypoint_scheme).to(device)
        if os.path.exists(dec_path):
            hm_decoder.load_state_dict(load_file(dec_path, device=str(device)), strict=True)
        hm_decoder = hm_decoder.to(dtype)
        
        noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder='scheduler')

        sdpose_model = {
            "unet": unet,
            "vae": vae,
            "empty_text_embed": empty_text_embed,
            "decoder": hm_decoder,
            "scheduler": noise_scheduler,
            "keypoint_scheme": keypoint_scheme,
            "device": device,
            "unload_on_finish": unload_on_finish
        }
        
        return (sdpose_model,)

# --- GroundingDINO模型加载 ---
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth",
    },
}

def list_groundingdino_model():
    return list(groundingdino_model_list.keys())

def get_bert_base_uncased_model_path():
    clip_model_base = os.path.join(folder_paths.models_dir, "clip")
    bert_path = os.path.join(clip_model_base, "bert-base-uncased")
    
    if os.path.exists(bert_path) and os.path.isdir(bert_path):
        has_bin = os.path.exists(os.path.join(bert_path, "pytorch_model.bin"))
        has_safe = os.path.exists(os.path.join(bert_path, "model.safetensors"))
        if has_bin or has_safe:
            print("SDPose Node: 使用 clip 目录下的 bert-base-uncased")
            return bert_path
    
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, "bert-base-uncased")
    if glob.glob(os.path.join(comfy_bert_model_base, "**/model.safetensors"), recursive=True) or \
       glob.glob(os.path.join(comfy_bert_model_base, "**/pytorch_model.bin"), recursive=True):
        print("SDPose Node: 使用 models/bert-base-uncased")
        return comfy_bert_model_base
    
    print("SDPose Node: 使用 HuggingFace Hub 的 bert-base-uncased")
    return "bert-base-uncased"

def get_local_filepath(url, dirname, local_file_name=None):
    if not local_file_name:
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)

    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination and os.path.exists(destination):
        return destination

    folder = os.path.join(folder_paths.models_dir, dirname)
    os.makedirs(folder, exist_ok=True)

    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        from torch.hub import download_url_to_file
        print(f"SDPose Node: 下载 {url} 到 {destination}")
        download_url_to_file(url, destination)
    
    return destination

def load_groundingdino_model(model_name):
    try:
        from groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
        from groundingdino.models import build_model as local_groundingdino_build_model
        from groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
    except ImportError:
        raise ImportError("请安装 groundingdino-py: pip install groundingdino-py")

    dino_model_args = local_groundingdino_SLConfig.fromfile(
        get_local_filepath(
            groundingdino_model_list[model_name]["config_url"],
            "grounding-dino",
        ),
    )

    if dino_model_args.text_encoder_type == "bert-base-uncased":
        dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()

    dino = local_groundingdino_build_model(dino_model_args)
    checkpoint = torch.load(
        get_local_filepath(
            groundingdino_model_list[model_name]["model_url"],
            "grounding-dino",
        ), map_location="cpu"
    )
    dino.load_state_dict(
        local_groundingdino_clean_state_dict(checkpoint["model"]), strict=False
    )
    dino.eval()
    return dino

class GroundingDinoModelLoader_SDPose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_groundingdino_model(),),
            }
        }

    CATEGORY = "SDPose"
    FUNCTION = "main"
    RETURN_TYPES = ("GROUNDING_DINO_MODEL",)

    def main(self, model_name):
        dino_model = load_groundingdino_model(model_name)
        gd_model_wrapper = {
            "model": dino_model,
            "model_name": model_name,
        }
        return (gd_model_wrapper,)

# --- 核心处理器节点（优化：支持纯黑背景标准OpenPose姿态图独立输出）---
class SDPoseOODProcessor:
    class DetectionJob:
        def __init__(self, frame_idx, person_idx, input_tensor, crop_info):
            self.frame_idx = frame_idx
            self.person_idx = person_idx
            self.input_tensor = input_tensor
            self.crop_info = crop_info
            self.kpts = None
            self.scores = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sdpose_model": ("SDPOSE_MODEL",),
                "images": ("IMAGE",),
                "score_threshold": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.9, "step": 0.05}),
                "overlay_alpha": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "data_from_florence2": ("JSON",),
                "grounding_dino_model": ("GROUNDING_DINO_MODEL",),
                "prompt": ("STRING", {"default": "person .", "multiline": False}),
                "gd_threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
                "yolo_model": ("YOLO_MODEL",),
                "save_for_editor": ("BOOLEAN", {"default": False}),
                "filename_prefix_edit": ("STRING", {"default": "poses/pose_edit"}),
                "keep_face": ("BOOLEAN", {"default": True, "label_on": "保留面部", "label_off": "移除面部"}),
                "keep_hands": ("BOOLEAN", {"default": True, "label_on": "保留手部", "label_off": "移除手部"}),
                "keep_feet": ("BOOLEAN", {"default": True, "label_on": "保留脚部", "label_off": "移除脚部"}),
                "scale_for_xinsr": ("BOOLEAN", {"default": False, "label_on": "Xinsr缩放", "label_off": "默认缩放"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", "IMAGE")
    RETURN_NAMES = ("images", "pose_keypoint", "pure_black_openpose_images")
    FUNCTION = "process_sequence"
    CATEGORY = "SDPose"

    def process_sequence(
        self,
        sdpose_model,
        images,
        score_threshold,
        overlay_alpha,
        batch_size=8,
        grounding_dino_model=None,
        prompt="person .",
        gd_threshold=0.3,
        yolo_model=None,
        save_for_editor=False,
        filename_prefix_edit="poses/pose_edit",
        data_from_florence2=None,
        keep_face=True,
        keep_hands=True,
        keep_feet=True,
        scale_for_xinsr=False
    ):
        if not isinstance(sdpose_model, dict):
            raise ValueError("无效的SDPose模型格式")
        
        if not isinstance(images, torch.Tensor):
            images = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        
        if data_from_florence2 is None:
            data_from_florence2 = []
        
        device = sdpose_model.get("device", model_management.get_torch_device())
        keypoint_scheme = sdpose_model.get("keypoint_scheme", "body")
        input_size = (768, 1024)
        B, H, W, C = images.shape if len(images.shape) == 4 else (1, 512, 512, 3)
        # 修复：张量转numpy时强制裁剪到0-1范围，避免数据异常
        images_np = torch.clamp(images, 0.0, 1.0).cpu().numpy() if isinstance(images, torch.Tensor) else np.zeros((B, H, W, C))
        
        # 进度条初始化
        comfy_pbar = None
        progress = 0
        try:
            from comfy.utils import ProgressBar
            comfy_pbar = ProgressBar(B * 3)
        except ImportError:
            pass
        
        # 模型设备迁移
        print(f"SDPose Node: 确保模型在设备 {device} 上")
        try:
            for key in ["unet", "vae", "decoder"]:
                if key in sdpose_model and hasattr(sdpose_model[key], "to"):
                    current_device = next(sdpose_model[key].parameters()).device
                    if current_device != device:
                        print(f"SDPose Node: 迁移 {key} 从 {current_device} 到 {device}")
                        sdpose_model[key].to(device)
        except Exception as e:
            print(f"SDPose Node: 警告 - 部分模型迁移失败: {e}")
        
        # 实例化管道
        print("SDPose Node: 初始化推理管道")
        class MockPipeline:
            def __init__(self, model_dict):
                self.unet = model_dict.get("unet", None)
                self.vae = model_dict.get("vae", None)
                self.decoder = model_dict.get("decoder", None)
                self.scheduler = model_dict.get("scheduler", None)
                self.empty_text_embed = model_dict.get("empty_text_embed", None)
                self.device = model_dict.get("device", torch.device("cpu"))

            @torch.no_grad()
            def __call__(self, rgb_in, timesteps, test_cfg, **kwargs):
                if self.unet is None or self.vae is None or self.decoder is None:
                    raise ValueError("管道缺少核心模型")
                
                unet_dtype = self.unet.dtype
                bsz = rgb_in.shape[0]
                
                rgb_latent = self.vae.encode(rgb_in).latent_dist.sample() * 0.18215
                rgb_latent = rgb_latent.to(dtype=unet_dtype)
                t = torch.tensor(timesteps, device=self.device).long()
                text_embed = self.empty_text_embed.repeat((bsz, 1, 1)).to(dtype=unet_dtype) if self.empty_text_embed is not None else torch.zeros((bsz, 77, 768), device=self.device)
                task_emb_anno = torch.tensor([1, 0]).float().unsqueeze(0).to(self.device)
                task_emb_anno = torch.cat([torch.sin(task_emb_anno), torch.cos(task_emb_anno)], dim=-1).repeat(bsz, 1)
                task_emb_anno = task_emb_anno.to(dtype=unet_dtype)
                
                feat = self.unet(rgb_latent, t, text_embed, class_labels=task_emb_anno, return_dict=False, return_decoder_feats=True)
                return self.decoder.predict((feat,), None, test_cfg=test_cfg)
        
        if all(k in sdpose_model for k in ["unet", "vae", "decoder"]):
            pipeline = MockPipeline(sdpose_model)
        else:
            raise ValueError("SDPose模型不完整，缺少核心组件")
        
        # 检测人体
        print("SDPose Node: 阶段1/3 - 检测人体")
        all_jobs = []
        used_florence2 = False
        
        # 解析Florence2数据
        input_bboxes_per_frame_f2 = [[] for _ in range(B)]
        if data_from_florence2 and len(data_from_florence2) >= B:
            try:
                parsed_bboxes_list = []
                for i, frame_data in enumerate(data_from_florence2[:B]):
                    bboxes_for_frame = []
                    if isinstance(frame_data, dict) and 'bboxes' in frame_data:
                        for box in frame_data['bboxes']:
                            if isinstance(box, (list, tuple)) and len(box) == 4:
                                try:
                                    bboxes_for_frame.append([float(coord) for coord in box])
                                except:
                                    continue
                    parsed_bboxes_list.append(bboxes_for_frame)
                
                input_bboxes_per_frame_f2 = parsed_bboxes_list
                used_florence2 = True
                print("SDPose Node: 使用Florence2提供的边界框")
            except Exception as e:
                print(f"SDPose Node: 警告 - Florence2数据解析失败: {e}")
        
        # 处理每帧
        for frame_idx in range(B):
            try:
                # 修复3：强制裁剪数据到0-255，避免溢出导致全黑
                original_image_rgb = (images_np[frame_idx] * 255).astype(np.float32)
                original_image_rgb = np.clip(original_image_rgb, 0, 255).astype(np.uint8) if len(images_np) > frame_idx else np.ones((H, W, C), dtype=np.uint8) * 255
                original_image_bgr = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR) if C == 3 else original_image_rgb
                frame_H, frame_W = original_image_rgb.shape[:2]
                bboxes = []
                
                # 优先级1: Florence2
                if used_florence2 and input_bboxes_per_frame_f2[frame_idx]:
                    bboxes = input_bboxes_per_frame_f2[frame_idx]
                
                # 优先级2: GroundingDINO
                if not bboxes and grounding_dino_model is not None:
                    if isinstance(prompt, str) and prompt.strip():
                        try:
                            pil_image = Image.fromarray(original_image_rgb)
                            gd_bboxes = groundingdino_predict(grounding_dino_model, pil_image, prompt, gd_threshold)
                            if gd_bboxes:
                                bboxes = gd_bboxes
                        except Exception as e:
                            print(f"SDPose Node: 帧{frame_idx} GroundingDINO预测失败: {e}")
                
                # 优先级3: YOLO
                if not bboxes and yolo_model is not None and YOLO_AVAILABLE:
                    try:
                        results = yolo_model(original_image_bgr, verbose=False)
                        yolo_bboxes = []
                        for result in results:
                            if result.boxes is not None:
                                for box in result.boxes:
                                    if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                                        yolo_bboxes.append(box.xyxy[0].cpu().numpy().tolist())
                        if yolo_bboxes:
                            bboxes = yolo_bboxes
                    except Exception as e:
                        print(f"SDPose Node: 帧{frame_idx} YOLO预测失败: {e}")
                
                # 优先级4: 整图
                if not bboxes:
                    bboxes = [[0, 0, frame_W, frame_H]]
                
                # 创建作业
                for person_idx, bbox in enumerate(bboxes):
                    try:
                        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                            continue
                        x1, y1, x2, y2 = map(float, bbox)
                        if not (x2 > x1 and y2 > y1):
                            continue
                        
                        input_tensor, _, crop_info = preprocess_image_for_sdpose(original_image_bgr, bbox, input_size)
                        all_jobs.append(self.DetectionJob(frame_idx, person_idx, input_tensor, crop_info))
                    except Exception as e:
                        continue
            except Exception as e:
                print(f"SDPose Node: 帧{frame_idx}处理失败: {e}")
            
            progress += 1
            if comfy_pbar:
                comfy_pbar.update_absolute(progress)
        
        total_detections = len(all_jobs)
        print(f"SDPose Node: 阶段1完成，检测到 {total_detections} 个人体")
        
        # 批量推理
        print(f"SDPose Node: 阶段2/3 - 批量推理 (批次大小: {batch_size})")
        for i in range(0, total_detections, batch_size):
            batch_jobs = all_jobs[i : i + batch_size]
            current_batch_size = len(batch_jobs)
            
            if current_batch_size == 0:
                continue
            
            try:
                batch_tensors = torch.cat([job.input_tensor for job in batch_jobs], dim=0).to(device)
                
                with torch.no_grad():
                    out = pipeline(batch_tensors, timesteps=[999], test_cfg={'flip_test': False}, show_progress_bar=False, mode="inference")
                
                for j in range(current_batch_size):
                    if j < len(out):
                        batch_jobs[j].kpts = out[j].keypoints[0] if hasattr(out[j], 'keypoints') else np.array([])
                        batch_jobs[j].scores = out[j].keypoint_scores[0] if hasattr(out[j], 'keypoint_scores') else np.array([])
            except Exception as e:
                print(f"SDPose Node: 批次 {i//batch_size} 处理失败: {e}")
            
            progress_update = (current_batch_size / total_detections) * B
            progress += progress_update
            if comfy_pbar:
                comfy_pbar.update_absolute(int(progress) + B)
        
        print(f"SDPose Node: 阶段2完成，处理 {total_detections} 个人体")
        
        # 重组帧
        print("SDPose Node: 阶段3/3 - 重组帧")
        frame_data = []
        pure_black_pose_images = []  # 存储纯黑背景标准OpenPose姿态图
        
        for i in range(B):
            original_rgb = (images_np[i] * 255).astype(np.float32)
            original_rgb = np.clip(original_rgb, 0, 255).astype(np.uint8) if len(images_np) > i else np.ones((H, W, C), dtype=np.uint8) * 255
            frame_H, frame_W = original_rgb.shape[:2]
            # 优化：OpenPose标准纯黑画布初始化
            openpose_black_canvas = np.zeros((frame_H, frame_W, 3), dtype=np.uint8)
            openpose_black_canvas = np.ascontiguousarray(openpose_black_canvas)
            frame_data.append({
                "canvas": openpose_black_canvas,
                "all_keypoints": [],
                "all_scores": [],
            })
        
        # 处理作业结果
        for job in all_jobs:
            try:
                frame_idx = job.frame_idx
                if frame_idx >= len(frame_data) or job.kpts is None or job.scores is None:
                    continue
                
                kpts_original = restore_keypoints_to_original(
                    job.kpts, job.crop_info, input_size, (W, H)
                )
                scores = job.scores
                kpts_final = kpts_original.copy()
                scores_final = scores.copy()
                
                # 绘制姿态
                if keypoint_scheme == "body":
                    frame_data[frame_idx]["canvas"] = draw_body17_keypoints_openpose_style(
                        frame_data[frame_idx]["canvas"], kpts_final, scores_final,
                        threshold=score_threshold, scale_for_xinsr=scale_for_xinsr
                    )
                else:
                    # 133->134点转换
                    if len(kpts_original) >= 17:
                        neck = (kpts_original[5] + kpts_original[6]) / 2
                        neck_score = min(scores[5], scores[6]) if scores[5] > 0.3 and scores[6] > 0.3 else 0
                        kpts_final = np.insert(kpts_original, 17, neck, axis=0)
                        scores_final = np.insert(scores, 17, neck_score)
                        
                        mmpose_idx = np.array([17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3])
                        openpose_idx = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17])
                        
                        temp_kpts = kpts_final.copy()
                        temp_scores = scores_final.copy()
                        temp_kpts[openpose_idx] = kpts_final[mmpose_idx]
                        temp_scores[openpose_idx] = scores_final[mmpose_idx]
                        
                        kpts_final = temp_kpts
                        scores_final = temp_scores
                    
                    # 移除指定部位
                    if len(kpts_final) >= 134 and len(scores_final) >= 134:
                        if not keep_face:
                            kpts_final[24:92] = 0.0
                            scores_final[24:92] = 0.0
                        if not keep_hands:
                            kpts_final[92:113] = 0.0
                            scores_final[92:113] = 0.0
                            kpts_final[113:134] = 0.0
                            scores_final[113:134] = 0.0
                        if not keep_feet:
                            kpts_final[18:24] = 0.0
                            scores_final[18:24] = 0.0
                    
                    # 绘制全身姿态
                    frame_data[frame_idx]["canvas"] = draw_wholebody_keypoints_openpose_style(
                        frame_data[frame_idx]["canvas"], kpts_final, scores_final,
                        threshold=score_threshold, scale_for_xinsr=scale_for_xinsr
                    )
                
                # 保存关键点
                frame_data[frame_idx]["all_keypoints"].append(kpts_final)
                frame_data[frame_idx]["all_scores"].append(scores_final)
            except Exception as e:
                continue
        
        # 生成结果
        result_images = []
        all_frames_json_data = []
        
        for frame_idx in range(B):
            try:
                # 原有代码：获取原图和姿态画布
                original_image_rgb = (images_np[frame_idx] * 255).astype(np.float32)
                original_image_rgb = np.clip(original_image_rgb, 0, 255).astype(np.uint8) if len(images_np) > frame_idx else np.ones((H, W, C), dtype=np.uint8) * 255
                pose_canvas = frame_data[frame_idx]["canvas"]
                
                # 新增：纯黑背景姿态图标准化（直接获取绘制后的canvas，不叠加原图）
                pure_black_pose = np.clip(pose_canvas, 0, 255).astype(np.uint8)
                pure_black_pose_images.append(pure_black_pose)  # 保存纯黑背景姿态图
                
                # 校验纯黑背景姿态图（可选，用于调试）
                if frame_idx == 0:
                    is_valid, msg = validate_openpose_black_image(pure_black_pose)
                    print(f"SDPose Node: 纯黑背景姿态图校验结果（帧{frame_idx}）：{msg}")
                
                # 原有代码：叠加原图（保留原有功能，可选）
                overlay_alpha = np.clip(overlay_alpha, 0.0, 1.0)
                result_image = cv2.addWeighted(original_image_rgb, 1.0 - overlay_alpha, pose_canvas, overlay_alpha, 0.0)
                result_image = np.clip(result_image, 0, 255).astype(np.uint8)
                result_images.append(result_image)
                
                # 生成JSON
                frame_json = convert_to_openpose_json(
                    frame_data[frame_idx]["all_keypoints"],
                    frame_data[frame_idx]["all_scores"],
                    W, H, keypoint_scheme
                )
                all_frames_json_data.append(frame_json)
                
                # 保存编辑器JSON
                if save_for_editor:
                    data_to_save = convert_to_loader_json(
                        frame_data[frame_idx]["all_keypoints"],
                        frame_data[frame_idx]["all_scores"],
                        W, H, keypoint_scheme, score_threshold
                    )
                    
                    output_dir = folder_paths.get_output_directory()
                    is_batch = B > 1
                    
                    if is_batch:
                        filename_to_use = f"{filename_prefix_edit}_frame{frame_idx:06d}"
                        full_output_folder, base_filename, _, _, _ = folder_paths.get_save_image_path(filename_to_use, output_dir, W, H)
                    else:
                        full_output_folder, base_filename, _, _, _ = folder_paths.get_save_image_path(filename_prefix_edit, output_dir, W, H)
                        
                        # 查找计数器
                        max_num = -1
                        pattern = re.compile(re.escape(base_filename) + r"_(\d{5,})\.json")
                        try:
                            for f in os.listdir(full_output_folder):
                                match = pattern.match(f)
                                if match:
                                    num = int(match.group(1))
                                    max_num = max(max_num, num)
                        except FileNotFoundError:
                            pass
                        
                        new_counter = max_num + 1
                        base_filename = f"{base_filename}_{new_counter:05d}"
                    
                    # 保存文件
                    os.makedirs(full_output_folder, exist_ok=True)
                    final_filename = f"{base_filename}.json"
                    file_path = os.path.join(full_output_folder, final_filename)
                    
                    with open(file_path, 'w') as f:
                        json.dump(data_to_save, f, indent=4)
                    
                    if not is_batch and frame_idx == 0:
                        print(f"SDPose Node: 保存JSON到: {file_path}")
            except Exception as e:
                print(f"SDPose Node: 帧{frame_idx}重组失败: {e}")
            
            progress += 1
            if comfy_pbar:
                comfy_pbar.update_absolute(progress + B * 2)
        
        print("SDPose Node: 阶段3完成，重组帧结束")
        
        # 模型卸载
        if sdpose_model.get("unload_on_finish", False):
            print("SDPose Node: 卸载模型到CPU")
            offload_device = torch.device("cpu")
            for key in ["unet", "vae", "decoder"]:
                if key in sdpose_model and hasattr(sdpose_model[key], 'to'):
                    try:
                        sdpose_model[key].to(offload_device)
                    except Exception as e:
                        print(f"SDPose Node: 警告 - 卸载{key}失败: {e}")
            model_management.soft_empty_cache()
        
        # 修复7：最终张量转换时强制归一化到0-1范围，避免数据异常
        if result_images:
            result_np = np.stack(result_images, axis=0).astype(np.float32) / 255.0
            result_tensor = torch.from_numpy(np.clip(result_np, 0.0, 1.0))
        else:
            result_tensor = torch.clamp(torch.from_numpy(images_np.astype(np.float32)), 0.0, 1.0)
        
        # 转换纯黑背景姿态图为张量（标准化到0-1范围）
        if pure_black_pose_images:
            pure_black_pose_np = np.stack(pure_black_pose_images, axis=0).astype(np.float32) / 255.0
            pure_black_pose_tensor = torch.from_numpy(np.clip(pure_black_pose_np, 0.0, 1.0))
        else:
            pure_black_pose_tensor = torch.zeros_like(result_tensor)
        
        # 返回结果（包含纯黑背景姿态图）
        return (result_tensor, all_frames_json_data, pure_black_pose_tensor)

# --- ComfyUI节点映射 ---
NODE_CLASS_MAPPINGS = {
    "SDPoseOODLoader": SDPoseOODLoader,
    "SDPoseOODProcessor": SDPoseOODProcessor,
    "YOLOModelLoader": YOLOModelLoader,
    "GroundingDinoModelLoader_SDPose": GroundingDinoModelLoader_SDPose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDPoseOODLoader": "Load SDPose Model",
    "SDPoseOODProcessor": "Run SDPose Estimation",
    "YOLOModelLoader": "Load YOLO Model",
    "GroundingDinoModelLoader_SDPose": "Load GroundingDINO Model (SDPose)",
}