from internal import image
from internal import utils
import jax
import dm_pix

high_path1 = '/data/xielangren/project/LLNeRF-main/datasets/llnerf-dataset/still4/high/DSC01654.JPG'
high_path2 = '/data/xielangren/project/LLNeRF-main/datasets/llnerf-dataset/still4/high/DSC01670.JPG'
high_path3 = '/data/xielangren/project/LLNeRF-main/datasets/llnerf-dataset/still4/high/DSC01686.JPG'


enhanced_path1 = '/data/xielangren/project/LLNeRF-main/nerf_results/llnerf/llnerf__still4/render/test_preds_step_105000/rgb_enhanced_000.png'
enhanced_path2 = '/data/xielangren/project/LLNeRF-main/nerf_results/llnerf/llnerf__still4/render/test_preds_step_105000/rgb_enhanced_001.png'
enhanced_path3 = '/data/xielangren/project/LLNeRF-main/nerf_results/llnerf/llnerf__still4/render/test_preds_step_105000/rgb_enhanced_002.png'


avg_psnr = 0
avg_ssim = 0
ssim_fn = jax.jit(dm_pix.ssim)

high_image1 = utils.load_img(high_path1) / 255.
high_image2 = utils.load_img(high_path2) / 255.
high_image3 = utils.load_img(high_path3) / 255.


enhanced_image1 = utils.load_img(enhanced_path1) / 255.
enhanced_image2 = utils.load_img(enhanced_path2) / 255.
enhanced_image3 = utils.load_img(enhanced_path3) / 255.

psnr1 = float(image.mse_to_psnr(((enhanced_image1[:,:-1,:] - high_image1)**2).mean()))
ssim1 = float(ssim_fn(enhanced_image1[:,:-1,:], high_image1))
print(f'psnr1: {psnr1 :0.6f}, ssim1:{ssim1:0.6f}')

psnr2 = float(image.mse_to_psnr(((enhanced_image2[:,:-1,:] - high_image2)**2).mean()))
ssim2 = float(ssim_fn(enhanced_image2[:,:-1,:], high_image2))
print(f'psnr2: {psnr2 :0.6f}, ssim2:{ssim2:0.6f}')

psnr3 = float(image.mse_to_psnr(((enhanced_image3[:,:-1,:] - high_image3)**2).mean()))
ssim3 = float(ssim_fn(enhanced_image3[:,:-1,:], high_image3))
print(f'psnr3: {psnr3 :0.6f}, ssim3:{ssim3:0.6f}')


avg_psnr = (psnr1+psnr2+psnr3)/3.0
avg_ssim = (ssim1+ssim2+ssim3)/3.0
print((f'avg_psnr:{avg_psnr:0.6f},avg_ssim:{avg_ssim:0.6f}'))