## ⌨️ How to run

1. ⚙️ **Setup the environment:** We provide the exported conda yaml file `environment.yml`. Please make sure you installed `conda` and run:

  ```shell
  conda env create -f environment.yml
  conda activate llnerf
  ```
  Note that this repo requires `jax` and `flax`. We use `cuda 11.7` and `cudnn 8.2`. If you need to set up the Python environment with a different version of cuda+cudnn, we suggest you manually install jax, jaxlib, and flax to ensure compatibility with your cuda environment. Please refer to their official documentation for installation instructions. If you encounter any issues during the jax installation, please consult their official documentation for troubleshooting.


2. 📂 **Download the dataset:** dataset is [here](https://drive.google.com/drive/folders/1h-u8DkvuaIvcHZihYIWcqwpURiM32_u3?usp=sharing). Please download and unzip it.

3. 🏃 **Training:** Please modify `scripts/train.sh` first by replacing the dataset path and scene name with yours, and run `bash scripts/train.sh`.

4. 🎥 **Rendering:** Please modify `scripts/render.sh` first by replacing the dataset path and scene name with yours, and run `bash scripts/render.sh`.

## 🔗 Cite This Paper

```bibtex
@inproceedings{wang2023lighting,
  title={Lighting up NeRF via Unsupervised Decomposition and Enhancement},
  author={Haoyuan Wang, Xiaogang Xu, Ke Xu, and Rynson W.H. Lau},
  booktitle={ICCV},
  year={2023}
}
```

