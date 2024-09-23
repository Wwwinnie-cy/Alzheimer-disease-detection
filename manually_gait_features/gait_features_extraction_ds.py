import json
import numpy as np
import os
import matplotlib.pyplot as plt

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_data(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
        print("Data saved to", file_path)

def calculate_velocity(positions, frame_rate):
    velocities = []
    for i in range(1, len(positions)):
        velocity = (np.array(positions[i]) - np.array(positions[i-1])) * frame_rate
        velocities.append(velocity)
    return velocities

def calculate_acceleration(velocities, frame_rate):
    accelerations = []
    for i in range(1, len(velocities)):
        acceleration = (velocities[i] - velocities[i-1]) * frame_rate
        accelerations.append(acceleration)
    return accelerations

def classify_gait_phases(left_heel_positions, right_heel_positions, frame_rate, group):
    left_velocities = calculate_velocity(left_heel_positions, frame_rate)
    right_velocities = calculate_velocity(right_heel_positions, frame_rate)
    
    left_accelerations = calculate_acceleration(left_velocities, frame_rate)
    right_accelerations = calculate_acceleration(right_velocities, frame_rate)
    
    stance_times = 0
    swing_times = 0
    double_support_times = 0

    if group == "AD":
        velocity_threshold = 1.138  # pixels per second
        acceleration_threshold = 1062
    elif group == "HC":
        velocity_threshold = 1.089  # pixels per second
        acceleration_threshold = 1295      
    
    for i in range(1, len(left_accelerations)):
        if np.linalg.norm(left_velocities[i]) < velocity_threshold and np.linalg.norm(right_velocities[i]) < velocity_threshold:
            stance_times += 1 / frame_rate
            if np.linalg.norm(left_accelerations[i]) < acceleration_threshold and np.linalg.norm(right_accelerations[i]) < acceleration_threshold:
                double_support_times += 1 / frame_rate
        else:
            swing_times += 1 / frame_rate
    
    return {
        "stance_times": stance_times,
        "swing_times": swing_times,
        "double_support_times": double_support_times
    }

def update_and_save_new_features(keypoints_directory, new_features_file, frame_rate):
    new_features_data = {}
    
    for filename in os.listdir(keypoints_directory):
        if filename.endswith('.json'):
            keypoints_file = os.path.join(keypoints_directory, filename)
            with open(keypoints_file, 'r') as file:
                data = json.load(file)
            
            left_heel_positions = [frame['instances'][0]['keypoints'][15] for frame in data]
            right_heel_positions = [frame['instances'][0]['keypoints'][16] for frame in data]
            
            results = classify_gait_phases(left_heel_positions, right_heel_positions, frame_rate, 'HC')

            file_key = os.path.splitext(filename)[0]
            new_features_data[file_key] = results

    save_data(new_features_data, new_features_file)

frame_rate = 30  # fps
keypoints_directory = 'D:\\是大学啦\\AD\\dual_task_gait\\OutputResults\\HC_keypoints'
new_features_file = 'D:\\是大学啦\\AD\\dual_task_gait\\OutputResults\\HC_gait_features_ds.json'
update_and_save_new_features(keypoints_directory, new_features_file, frame_rate)

def plot_heel_positions(keypoints_directory):
    left_heel_y_positions = []
    right_heel_y_positions = []

    for filename in os.listdir(keypoints_directory):
        if filename.endswith('.json'):
            keypoints_file = os.path.join(keypoints_directory, filename)
            data = load_data(keypoints_file)
            
            # 收集左脚和右脚的y坐标
            for frame in data:
                left_heel_y = frame['instances'][0]['keypoints'][15][1]  # 假设y坐标是列表的第二个元素
                right_heel_y = frame['instances'][0]['keypoints'][16][1]
                left_heel_y_positions.append(left_heel_y)
                right_heel_y_positions.append(right_heel_y)

    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(left_heel_y_positions)), left_heel_y_positions, c='blue', label='Left Heel Y Position', alpha=0.5)
    #plt.scatter(range(len(right_heel_y_positions)), right_heel_y_positions, c='red', label='Right Heel Y Position', alpha=0.5)
    plt.title('Heel Y Positions Over Frames')
    plt.xlabel('Frame Index')
    plt.ylabel('Y Position')
    plt.legend()
    plt.show()

#plot_heel_positions(keypoints_directory)

