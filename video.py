import subprocess

# 输入的PNG图像文件名格式
input_pattern = '/data/xielangren/project/LLNeRF-main/nerf_results/llnerf/llnerf__moto_s3im/render/path_renders_step_105000/rgb_enhanced_%03d.png'  # 例如：input0001.png, input0002.png, ...

# 输出的MP4视频文件名
output_file = '/data/xielangren/project/LLNeRF-main/nerf_results/llnerf/llnerf__moto_s3im/render/enhanced.mp4'

# 使用FFmpeg命令将PNG图像转换为MP4视频
ffmpeg_cmd = [
    'ffmpeg',
    '-framerate', '30',  # 帧率，根据需要进行调整
    '-i', input_pattern,  # 输入图像文件名格式
    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  
    '-c:v', 'libx264',    # 使用x264视频编解码器
    '-crf', '18',         # 视频质量，18是默认值，根据需要进行调整
    '-pix_fmt', 'yuv420p', # 像素格式
    output_file
]

subprocess.run(ffmpeg_cmd)

print(f'转换完成，输出文件：{output_file}')