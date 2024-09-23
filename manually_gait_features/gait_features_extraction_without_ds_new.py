import subprocess
import json
import numpy as np
import os
from moviepy.editor import VideoFileClip

def get_frame_rate(video_path):
    """获取视频的帧率"""
    cmd = f"ffmpeg -i {video_path}"
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    lines = result.stdout.split('\n')
    for line in lines:
        if "fps" in line:
            # 解析帧率
            parts = line.split(',')
            for part in parts:
                if "fps" in part:
                    fps_info = part.strip()
                    fps = float(fps_info.split(' ')[0])
                    return fps
    return None  

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def calculate_distance(p1, p2):
    """计算两点之间的距离"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_distance_hor(p1, p2):
    """计算两点之间的距离"""
    return abs(p1[0] - p2[0])

def calculate_step_length(left_heel_prev, right_heel_prev, left_heel_curr, right_heel_curr):
    """计算步长, 单位为像素"""
    step_length_right = calculate_distance(left_heel_prev, right_heel_curr)
    step_length_left = calculate_distance(right_heel_prev, left_heel_curr)
    return step_length_left, step_length_right

def calculate_stride_length(left_heel_prev, right_heel_prev, left_heel_curr, right_heel_curr):
    """计算步幅, 单位为像素"""
    stride_length_left = calculate_distance(left_heel_prev, left_heel_curr)
    stride_length_right = calculate_distance(right_heel_prev, right_heel_curr)
    return stride_length_left, stride_length_right

def calculate_cadence(stride_time):
    """计算步频，单位：步/分钟"""
    return 60 / stride_time if stride_time > 0 else 0
    
def calculate_step_time(step_lengths, walking_speed):
    """计算步长时间，单位为秒"""
    step_times = []
    
    for step_length in step_lengths:
        step_time = step_length / walking_speed  # 步长 / 行走速度
        step_times.append(step_time)
    
    average_step_time = np.mean(step_times)
    step_time_variability = np.std(step_times)
    return average_step_time, step_time_variability

def calculate_stride_time(stride_lengths, walking_speed):
    "计算步幅时间，单位为秒"
    stride_times = []
    
    for stride_length in stride_lengths:
        stride_time = stride_length / walking_speed  # 步幅 / 行走速度
        stride_times.append(stride_time)
    
    average_stride_time = np.mean(stride_times)
    stride_time_variability = np.std(stride_times)
    return average_stride_time, stride_time_variability

def calculate_step_length_variability(step_lengths):
    """计算步长变异度"""
    step_length_variability = np.std(step_lengths)  # 步长的标准差，反映步长的波动性和空间变异性
    return step_length_variability

def calculate_stride_length_variability(stride_lengths):
    """计算步幅变异度"""
    stride_length_variability = np.std(stride_lengths)  # 步幅的标准差，反映步幅的波动性和空间变异性
    return stride_length_variability

def calculate_asymmetry(left_value, right_value):
    """计算不对称性，计算结果是百分比"""
    average = (left_value + right_value) / 2
    return abs(left_value - right_value) / average * 100 if average != 0 else 0

def calculate_walking_speed(video_path, distance_meters):
    """
    行走速度，单位为米/秒。
    """
    # 使用 moviepy 读取视频时长
    try:
        with VideoFileClip(video_path) as clip:
            total_seconds = clip.duration  # 获取视频的总时长，单位为秒
    except Exception as e:
        print(f"无法加载视频文件: {e}")
        return None

    # 计算行走速度
    if total_seconds > 0:  # 确保时间大于0，避免除以零
        walking_speed = distance_meters / total_seconds
        return walking_speed
    else:
        raise ValueError("从视频文件中获取的总时间必须大于0")


def has_full_stride_occurred(last_heel, current_heel, ground_threshold=570, tolerance=10, min_x_movement=8):
    """
    检查是否完成了一个完整的步态周期。
    
    参数:
    last_heel (tuple): 上一帧中脚后跟的 (x, y) 坐标。
    current_heel (tuple): 当前帧中脚后跟的 (x, y) 坐标。
    ground_threshold (int): 判断脚是否接触地面的像素阈值（越大表示越靠近地面）。
    tolerance (int): 判断两个位置是否接近的容忍度（像素值），用于判断同一只脚是否完成一个步态周期。
    min_x_movement (int): 判断 x 轴方向上是否有足够的移动，避免微小移动导致的误判。
    
    返回值:
    bool: 如果完成了一个完整的步态周期，则返回 True；否则返回 False。
    """
    # 检查当前脚后跟是否在地面（y 轴位置 >= 阈值）
    if current_heel[1] >= ground_threshold:
        # 检查当前脚后跟位置是否与上一帧的脚后跟位置足够接近
        if abs(current_heel[1] - last_heel[1]) <= tolerance:
            # 确保在 x 轴方向上有足够的移动
            if abs(current_heel[0] - last_heel[0]) >= min_x_movement:
                return True
    
    return False


# 计算步长和步态长度及相关指标
def calculate_step_and_stride_lengths(data, frame_rate, walking_speed):
    step_lengths = []
    stride_lengths = []
    step_asymmetry = []
    stride_asymmetry = []

    last_left_heel = None
    last_right_heel = None
    initial_left_heel = None
    initial_right_heel = None

    # 用于累积步态周期内的移动距离
    accumulated_left_stride_distance = 0
    accumulated_step_distance = 0

    for i in range(1, len(data)):
        if data[i-1]['instances'] and data[i]['instances']:
            keypoints_prev = data[i-1]['instances'][0]['keypoints']
            keypoints_curr = data[i]['instances'][0]['keypoints']

            left_heel_prev = keypoints_prev[15]
            right_heel_prev = keypoints_prev[16]
            left_heel_curr = keypoints_curr[15]
            right_heel_curr = keypoints_curr[16]

            if last_left_heel is None:
                last_left_heel = left_heel_prev
                initial_left_heel = left_heel_prev
                initial_right_heel = right_heel_prev

            # 累积左脚在整个周期内的移动距离
            accumulated_left_stride_distance += calculate_distance_hor(last_left_heel, left_heel_curr)

            # 累积当前步长的移动距离（右脚移动）
            if last_right_heel is not None:
                accumulated_step_distance += calculate_distance_hor(last_right_heel, right_heel_curr)

            # 检查是否完成了一个完整的步态周期
            if has_full_stride_occurred(last_left_heel, left_heel_curr):
                # 计算步态长度（stride length）
                stride_length_left = accumulated_left_stride_distance
                stride_lengths.append(stride_length_left)
                print(f'left stride length {stride_length_left}')

                # 计算步长（step length），即累积的右脚移动距离
                step_length_left = accumulated_step_distance / 2
                step_lengths.append(step_length_left)
                print(f'left step length {step_length_left}')

                # 如果上一个周期已经有右脚的步态长度，计算不对称性
                if last_right_heel is not None and initial_right_heel is not None:
                    stride_length_right = calculate_distance_hor(initial_right_heel, right_heel_curr)

                    # 计算步态不对称性
                    stride_asymmetry_value = calculate_asymmetry(stride_length_left, stride_length_right)
                    stride_asymmetry.append(stride_asymmetry_value)

                    # 计算步长不对称性
                    step_asymmetry_value = calculate_asymmetry(step_length_left, stride_length_right / 2)
                    step_asymmetry.append(step_asymmetry_value)

                # 重置累积器
                accumulated_left_stride_distance = 0  # 重置累积距离
                accumulated_step_distance = 0

                # 更新步态周期的初始位置
                initial_left_heel = left_heel_curr
                initial_right_heel = right_heel_curr
                last_right_heel = right_heel_curr
                last_left_heel = left_heel_curr
            else:
                last_left_heel = left_heel_curr
                last_right_heel = right_heel_curr

    avg_stride_time, stride_time_variability = calculate_stride_time(stride_lengths, walking_speed)
    avg_step_time, step_time_variability = calculate_step_time(step_lengths, walking_speed)

    return step_lengths, stride_lengths, avg_step_time, step_time_variability, avg_stride_time, stride_time_variability, step_asymmetry, stride_asymmetry


def calculate_gait_parameters(data, frame_rate, video_path):
    walking_speed = calculate_walking_speed(video_path, 6)
    step_lengths, stride_lengths, avg_step_time, step_time_variability, avg_stride_time, stride_time_variability, step_asymmetry, stride_asymmetry = calculate_step_and_stride_lengths(data, frame_rate, walking_speed)

    return compile_results(step_lengths, stride_lengths, avg_step_time, step_time_variability, avg_stride_time, stride_time_variability, step_asymmetry, stride_asymmetry, walking_speed)



def compile_results(step_lengths, stride_lengths, avg_step_time, step_time_variability, avg_stride_time, stride_time_variability, step_asymmetry, stride_asymmetry, walking_speed):
    return {
        "average_step_length": np.mean(step_lengths),
        "average_stride_length": np.mean(stride_lengths),
        "step_length_variability": np.std(step_lengths),
        "stride_length_variability": np.std(stride_lengths),
        "average_step_time": avg_step_time,
        "average_stride_time": avg_stride_time,
        "step_time_variability": step_time_variability,
        "stride_time_variability": stride_time_variability,
        "average_step_assymetry": np.mean(step_asymmetry),
        "average_stride_assymetry": np.mean(stride_asymmetry),
        "walking_speed": walking_speed,
        "cadence": calculate_cadence(avg_stride_time),
    }


def process_videos(video_directory, json_directory, output_json_path):
    results = {}
    for filename in os.listdir(video_directory):
        if filename.endswith(".mp4"):
            video_path = os.path.join(video_directory, filename)
            frame_rate = get_frame_rate(video_path) 
            json_filename = filename.replace(".mp4", ".json")
            json_path = os.path.join(json_directory, json_filename)

            data = load_json_data(json_path) 
            gait_parameters = calculate_gait_parameters(data, frame_rate, video_path)
            results[filename] = gait_parameters

    with open(output_json_path, 'w') as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)

video_directory = "D:\\是大学啦\\AD\\dual_task_gait\\OutputResults\\AD_videos"
json_directory = 'D:\\是大学啦\\AD\\dual_task_gait\\OutputResults\\AD_keypoints'
output_json_path = 'D:\\是大学啦\\AD\\dual_task_gait\\OutputResults\\AD_gait_features_without_ds.json'

process_videos(video_directory, json_directory, output_json_path)