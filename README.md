<div align="center">
  <h1>🎮 Video Game Super-Resolution 🔍</h1>
</div>

<div align="center"><b>Eric Phann</b> | <b>Samantha Michael</b> | <b>Kidus Kidane</b></div>

<div align="center">DSBA 6165: Artificial Intelligence and Deep Learning</div>

<div align="center">Dr. Archit Parnami, Spring 2025</div>
<br>

<div align="center">
  <a href="https://github.com/dsba6010-llm-applications/AgenticRAG-CharlotteEatz/blob/main/docs/Final%20Project%20Report.pdf">📋 Final Report</a> |
  <a href="https://dinebot-uncc-dsba.streamlit.app/">🧑‍🏫 Presentation</a>
</div>
<br>

<p align="center">
  <img width="900" height="300" src="https://raw.githubusercontent.com/ericphann/video-game-super-resolution/refs/heads/main/Example.png">
</p>


## Objective
Modern video games are increasingly demanding on hardware. Technologies like Deep Learning Super Sampling (DLSS) allow games to run at lower native resolutions for better performance, then upscale the imagery to higher resolutions using neural networks. This technology enables gamers with a lower-end GPU to still enjoy newer releases and graphically intensive games.

This project explores deep learning approaches to image super-resolution specifically for video game images. Inspired by technologies like NVIDIA's DLSS, our research investigates which techniques best preserve the unique visual characteristics of video games when upscaling from lower to higher resolutions. Our findings could inform future development of more accessible super-resolution solutions for game graphics.


## Dataset
After thorough evaluation of multiple datasets (documented in our EDA folder), we selected the [**Super Resolution in Video Games Dataset (SRVG)**](https://www.kaggle.com/competitions/super-resolution-in-video-games/data) as our primary training data. This dataset consists of paired 270p and 1080p images from Unreal Engine projects, providing ideal content for training models specifically for game graphics upscaling.

We've processed and made this dataset available on Hugging Face at [**ericphann/video-game-super-resolution**](https://huggingface.co/datasets/ericphann/video-game-super-resolution/tree/main).


## Methods
1. **Convolutional Neural Networks (CNNs)**
   - Building upon pioneering work like SRCNN and VDSR
   - Focus on efficiency and performance through deep networks

2. **Generative Adversarial Networks (GANs)**
   - Inspired by innovative models such as SRGAN and ESRGAN
   - Emphasis on perceptual quality and realistic textures

3. **Transformers**
   - Exploring novel architectures like Swin2SR and SwinIR
   - Investigating attention mechanisms for detail preservation


## Conclusion
We built upon existing literature and research by successfully reproducing and adapting CNNs, GANs, and transformers for the specific task of video game image super-resolution. This was done through various approaches such as hyperparameter tuning, tweaking model architectures, and fine-tuning on the SRVG dataset.

While state-of-the-art, peak signal-to-noise ratio (PSNR) benchmarks were not reached, the results demonstrate the effectiveness of architectural tuning and optimization when applied to real-world datasets under constrained resources.

## References
Conde, M. V., Choi, U.-J., Burchi, M., & Timofte, R. (2022, September 22). Swin2SR: SwinV2 transformer for compressed image super-resolution and restoration. arXiv.org. https://arxiv.org/abs/2209.11345  

Dong, C., Loy, C. C., He, K., & Tang, X. (2014). Image super-resolution using deep convolutional networks. arXiv. https://arxiv.org/abs/1501.00092  

Kim, J., Lee, J. K., & Lee, K. M. (2016). Accurate image super-resolution using very deep convolutional networks. arXiv. https://arxiv.org/abs/1511.04587  

Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv. https://arxiv.org/abs/1409.1556  

Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Qiao, Y., & Change Loy, C. (2018). ESRGAN: Enhanced super-resolution generative adversarial networks. arXiv. https://arxiv.org/abs/1809.00219  

Wang, X., Xie, L., Dong, C., & Shan, Y. (2021). Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data. arXiv. https://arxiv.org/abs/2107.10833

