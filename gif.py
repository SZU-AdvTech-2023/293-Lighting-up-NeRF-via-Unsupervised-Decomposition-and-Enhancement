from PIL import Image
import os

def create_combined_gif(image_folder, gif_name, file_patterns):
    images = {pattern: [] for pattern in file_patterns}

    # 遍历文件夹中的文件
    for filename in sorted(os.listdir(image_folder)):
        for pattern in file_patterns:
            if filename.startswith(pattern) and (filename.endswith(".png") or filename.endswith(".jpg")):
                image_path = os.path.join(image_folder, filename)
                img = Image.open(image_path)
                images[pattern].append(img)

    # 确保每组图像的长度相同
    min_length = min(len(images[pattern]) for pattern in file_patterns)
    for pattern in file_patterns:
        images[pattern] = images[pattern][:min_length]

    # 合成为一个GIF，按2行2列排列
    combined_images = []

    for i in range(0, min_length, 2):
        combined_frame = Image.new("RGB", (img.width * 2, img.height * 2))

        for j, pattern in enumerate(file_patterns):
            combined_frame.paste(images[pattern][i], ((j % 2) * images[pattern][i].width, (j // 2) * images[pattern][i].height))

        combined_images.append(combined_frame)

    # 保存为GIF，每帧显示1秒/30帧
    combined_images[0].save(gif_name, save_all=True, append_images=combined_images[1:], duration=(1000 / 20), loop=0)


# 指定包含图片的文件夹和生成的GIF文件名
image_folder = "/data/xielangren/project/LLNeRF-main/nerf_results/llnerf/llnerf__bike/render/path_renders_step_105000/"
gif_name = "combined_output.gif"
file_patterns = ["color_", "R_norm_", "Lg22_", "rgb_enhanced_"]

# 创建合成的GIF
create_combined_gif(image_folder, gif_name, file_patterns)
