from PIL import Image
import os
from tqdm import tqdm

'''
由于部分的face_dataset的图片的格式是32位位深，但是在我的网络网络卷积中只有三层无法进行4层的卷积。

'''

def convert_to_24bit_depth(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取输入目录中的图片文件列表
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # 使用tqdm创建进度条
    for filename in tqdm(image_files, desc="Converting images", unit="image"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 打开图像
        image = Image.open(input_path)

        # 将图像转换为24位位深
        image = image.convert('RGB')

        # 保存转换后的图像
        image.save(output_path)

if __name__ == "__main__":
    # 指定输入和输出目录
    input_directory = 'face_dataset'
    output_directory = 'output'

    # 调用函数进行转换
    convert_to_24bit_depth(input_directory, output_directory)
