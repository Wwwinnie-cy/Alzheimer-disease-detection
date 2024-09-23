import os
import subprocess

def process_videos(input_dir, video_output_dir, pred_out_dir, progress_output_dir):
    # 读取进度文件，找出上次处理到哪个文件
    print('here?')
    progress_file = os.path.join(progress_output_dir, "progress_HC.txt")
    processed_files = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as file:
            processed_files = set(file.read().splitlines())

    print('hre?')
    # 列出指定目录下所有的.mp4文件
    for filename in os.listdir(input_dir):
        print(filename)
        if filename.endswith(".mp4") and filename not in processed_files:
            print(filename)
            video_path = os.path.join(input_dir, filename)
            command = [
                "D:/anaconda3/envs/mmpose/python.exe", "demo/inferencer_demo.py", video_path,
                "--pose2d", "human",
                "--draw-heatmap",
                "--vis-out-dir", video_output_dir,
                "--pred-out-dir", pred_out_dir
            ]
            print("Processing:", video_path)
            # 执行命令并捕获输出
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print(result.stdout)
                # 更新进度文件
                with open(progress_file, 'a') as file:
                    file.write(filename + "\n")
            except subprocess.CalledProcessError as e:
                print("Error occurred while processing", video_path)
                print("STDOUT:", e.stdout)
                print("STDERR:", e.stderr)

# 设置视频文件夹路径和输出路径
video_samples_dir = "D:\\是大学啦\\AD\\dual_task_gait\\depression\\Depression_zy_video"
video_output_results_dir = "D:\\是大学啦\\AD\\dual_task_gait\\depression\\Outputresults\\videos"
pred_output_dir = "D:\\是大学啦\\AD\\dual_task_gait\\depression\\Outputresults\\keypoints"
progress_output_dir = "D:\\是大学啦\\AD\\dual_task_gait\\depression\\Outputresults"
# 调用函数处理所有视频
process_videos(video_samples_dir, video_output_results_dir, pred_output_dir, progress_output_dir)
