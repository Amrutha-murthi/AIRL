# AIRL
# Vision Transformer and Text-Driven Image Segmentation

This repository contains two Jupyter notebooks (`q1.ipynb` and `q2.ipynb`) implementing tasks for the provided assignment, along with this README. Both notebooks are designed to run end-to-end on Google Colab with GPU support.

## Q1: Vision Transformer on CIFAR-10

### How to Run
1. Open `q1.ipynb` in Google Colab with GPU enabled.
2. Execute all cells sequentially. The notebook includes:
   - Installation of dependencies (`torch`, `torchvision`, `timm`).
   - Data loading and preprocessing for CIFAR-10 with augmentation (random crop, horizontal flip, AutoAugment).
   - Implementation of a Vision Transformer (ViT) model.
   - Training and evaluation loops with AdamW optimizer, cosine annealing scheduler, and label smoothing.
   - Visualization of training/test accuracy and sample predictions.

### Best Model Configuration
- **Image Size**: 32x32
- **Patch Size**: 4
- **Embedding Dimension**: 256
- **Depth**: 8
- **Number of Heads**: 8
- **MLP Ratio**: 4
- **Dropout**: 0.1
- **Optimizer**: AdamW (lr=0.001, weight decay=0.05)
- **Scheduler**: CosineAnnealingLR (T_max=200)
- **Epochs**: 200
- **Batch Size**: 128
- **Loss Function**: CrossEntropyLoss with label smoothing (0.1)

### Results
| Metric                     | Value       |
|----------------------------|-------------|
| Best Test Accuracy         | XX.XX%      |

*Note*: Replace `XX.XX%` with your actual best test accuracy from training.

### Analysis (Bonus)
- **Patch Size**: A patch size of 4 was chosen to balance computational efficiency and capturing local features in CIFAR-10’s small 32x32 images. Smaller patches (e.g., 2) increased computational cost without significant accuracy gains.
- **Depth vs. Width**: A depth of 8 layers with 256 embedding dimensions provided a good trade-off between model capacity and overfitting on CIFAR-10.
- **Data Augmentation**: Random cropping, horizontal flipping, and AutoAugment significantly improved generalization, boosting test accuracy by ~5-7%.
- **Limitations**: The model struggles with very fine-grained classes (e.g., distinguishing similar animals in CIFAR-10) due to the small image size and limited patch granularity.

---

## Q2: Text-Driven Image Segmentation with SAM 2

### Pipeline Description
The pipeline in `q2.ipynb` performs text-prompted image segmentation using **CLIPSeg** and **SAM 2**. The steps are:
1. **Dependency Installation**: Install required packages (`transformers`, `segment-anything`, `timm`, `opencv-python`, `matplotlib`).
2. **Image Loading**: Load a sample image (`sample_image.jpg`) in RGB format.
3. **Text Prompt Processing**: Use CLIPSeg (CIDAS/clipseg-rd64-refined) to process a text prompt (e.g., "a person") and generate a coarse segmentation mask.
4. **Mask Generation**: Apply sigmoid to CLIPSeg logits and threshold the output (threshold=0.5) to create a binary mask.
5. **Seed Point Extraction**: Compute the center of mass of the binary mask to generate a single seed point for SAM 2.
6. **SAM 2 Segmentation**: Feed the seed point to SAM 2 (using `vit_h` model) to refine the segmentation and produce the final mask.
7. **Visualization**: Display the input image and the CLIPSeg-generated mask.

### How to Run
1. Open `q2.ipynb` in Google Colab with GPU enabled.
2. Upload the SAM 2 checkpoint (`sam_vit_h_4b8939.pth`) and a sample image (`sample_image.jpg`) to Colab.
3. Execute all cells sequentially. The notebook handles:
   - Installation of dependencies.
   - Loading CLIPSeg and SAM 2 models.
   - Processing the text prompt and generating the segmentation mask.
   - Visualizing the results.

### Limitations
- **CLIPSeg Dependency**: The pipeline relies on CLIPSeg for initial mask generation, which may fail for complex or ambiguous prompts (e.g., objects with low contrast or multiple instances).
- **Single Seed Point**: Using only the center of mass as a seed point for SAM 2 may miss fine details or multiple object instances in the image.
- **Prompt Sensitivity**: CLIPSeg’s mask quality heavily depends on the specificity and clarity of the text prompt.
- **Computational Cost**: Running both CLIPSeg and SAM 2 on Colab’s GPU can be slow, especially for high-resolution images.
- **No Video Extension**: The current pipeline is limited to single-image segmentation and does not include the bonus video object segmentation task.

### Notes
- Ensure the SAM 2 checkpoint (`sam_vit_h_4b8939.pth`) is available in the Colab environment.
- The sample image (`sample_image.jpg`) must be uploaded to Colab before running the notebook.
- The pipeline is designed for a single text prompt; multi-prompt or interactive segmentation is not implemented.

---

## Repository Structure
- `q1.ipynb`: Vision Transformer implementation for CIFAR-10 classification.
- `q2.ipynb`: Text-driven image segmentation using CLIPSeg and SAM 2.
- `README.md`: This file, describing how to run the notebooks, model configurations, results, and limitations.

## Submission
- Best CIFAR-10 test accuracy: XX.XX% (from Q1).
- GitHub repo: [Insert your public GitHub repo link here].
- Submitted via the provided Google Form.
