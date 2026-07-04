# 🧊 ARCTiC ❄️
**A**utomated **R**emoval of **C**orrupted **T**ilts **i**n **C**ryo-ET

📄 Published in [*Journal of Structural Biology: X*](https://www.sciencedirect.com/science/article/pii/S259015242500011X)

## ⚙️ Installation

*   Get ARCTiC source codes

```
git clone https://github.com/turonova/ARCTiC
cd ARCTiC
```

### Using environment.yml

*   Create ARCTiC conda environment using .yml file

```
conda env create -f environment.yml
```

The environment is based on Python 3.10 and uses Conda (conda-forge channel) with additional pip packages.
CUDA-Enabled PyTorch: The environment uses torch==2.5.0+cu118 (CUDA 11.8).


### Manual installation

*   The code has the following dependencies:

1. `torch` – PyTorch for deep learning.
2. `torchvision` – Computer vision utilities (datasets, transforms, models).
3. `timm` – Pretrained models from `rwightman/pytorch-image-models`.
4. `cryocat` – Includes `cryomap` for handling cryo-ET images.
5. `numpy` – Numerical computing.
6. `matplotlib` – Visualization library for plots.
7. `tqdm` – Progress bars.
8. `scikit-learn` – Machine learning utilities, including classification reports and confusion matrices.
9. `pillow` – Image processing (`PIL.Image`).

## 🧠 Fine-tuned Models

*   Create a directory with fine-tuned models.

```
mkdir -p <models>
```

*   Download fine-tuned binary and multiclass models from 
    [ownCloud](https://oc.biophys.mpg.de/owncloud/s/zmMZPr2TEB4Bwda)
    and put them into `<models>`.


## 🚀 Usage

* To run the script from the command line, use the following syntax:

```
python run_TS_cleaning.py \
  --input_ts 'input_TS.mrc' \
  --cleaned_ts 'cleaned_TS.mrc' \
  --angle_start -50 \
  --angle_step 2 \
  --confidence_threshold 0.5 \
  --batch_size 16 \
  --pdf_output 'output_visualization.pdf' \
  --model 'models/swin_tiny_fine-tuned.pth' \
  --csv_output 'classification_results.csv' \
  --mdoc_input 'input_series.mdoc' \
  --mdoc_output 'cleaned_series.mdoc'
```

## 🧾 Arguments:
1. `--input_ts` `<path to input .mrc file>` (Required)
   - **Description:** Path to the input `.mrc` file, which contains the tilt series data to be processed.
   - **Example:** `'input_TS.mrc'`

2. `--cleaned_ts` `<path to output .mrc file>` (Required)
   - **Description:** Path to the output `.mrc` file where the cleaned tilt series will be saved.
   - **Example:** `'cleaned_TS.mrc'`

3. `--angle_start` `<float>` (Required)
   - **Description:** The starting tilt angle for visualizing tilt images. 
   - **Example:** `-50`

4. `--angle_step` `<float>` (Required)
   - **Description:** The increment (step size) for the tilt angles between consecutive tilts.
   - **Example:** `2`

5. `--confidence_threshold` `<float>` (Optional, default: `0.5`)
   - **Description:** The maximum allowable corruption probability before a tilt is excluded. This allows for simple, dual-direction control over cleaning aggressiveness:
      - **Higher thresholds (e.g., `0.8`)**: Very lenient. Slices are only thrown away if the model is at least 80% certain they are corrupted **(Excludes FEWER tilts)**.
      - **Lower thresholds (e.g., `0.2`)**: Very strict. Slices with even a 20% chance of being corrupted are thrown away **(Excludes MORE tilts)**.
   - **Example:** `0.5`

6. `--batch_size` `<int>` (Optional, default: `1`)
   - **Description:** The number of tilt images stacked and passed to the GPU simultaneously during evaluation. Increasing this accelerates processing significantly depending on your available VRAM.
   - **Example:** `16`

7. `--pdf_output` `<path to output PDF file>` (Optional, default: 'output_visualization.pdf')
   - **Description:** Path to the PDF file where the visualizations (tilt angles and excluded images with probability bars) will be saved.
   - **Example:** `'output_visualization.pdf'`

8. `--model` `<path to model file>` (Required)
   - **Description:** Path to the pre-trained model file (e.g., a Swin transformer model) that will be used for classifying images. The model should be compatible with the network architecture specified in the script.
   - **Example:** `'models/swin_tiny_fine-tuned.pth'`

9. `--csv_output` `<path to output CSV file>` (Optional)
   - **Description:** Path where the classification results CSV file will be saved. This CSV lists each slice with flags indicating if it should be removed. Providing this option enables export of slice classification results.
   - **Example:** `'classification_results.csv'`

10. `--mdoc_input` `<path to input .mdoc file>` (Optional)
   - **Description:** Path to the .mdoc metadata file associated with the tilt series. Used for removing metadata entries corresponding to corrupted tilts (as determined by the model).
   - **Example:** `'input_series.mdoc'`

11. `--mdoc_output` `<path to output .mdoc file>` (Optional)
   - **Description:** Output path for saving the cleaned .mdoc file. **Must be used together with --mdoc_input and --csv_output.**
   - **Example:** `'cleaned_series.mdoc'`


### This command will:

1. Load the input tilt series data from `input_TS.mrc`.
2. Pack frames into optimized inference batches of `16` to leverage high-performance GPU acceleration.
3. Use the `swin_tiny_fine_tuned.pth` model to clean TS and visualize tilt angles.
4. Apply your custom Corruption Threshold criteria limit to split good tilts from distorted ones.
5. Start tilt visualization at `-50` degrees with a step of `2` degrees.
6. Generate and save the visualizations (tilt angle and classification probability scale bars) into `output_visualization.pdf`.
7. Save the cleaned tilt series to `cleaned_TS.mrc`.
8. Export classification results (indices and probabilities) to a CSV file `classification_results.csv`.
9. If a corresponding `input_series.mdoc` file is provided, generate a cleaned version `cleaned_series.mdoc` by removing entries of excluded tilts.


## Additional Notes

- Ensure that the model file (`.pth`) is compatible with the architecture defined in the script (e.g., `swin_tiny` or `swin_large`).


## 📓 Jupyter Notebooks

- In the `notebooks` directory, there are additional Jupyter Notebooks that were used for augmentation (`augmentation.ipynb`), data split (`split_train_val_test.ipynb`), and examples of training and evaluation scripts.


## 📜 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

### APA format

Majtner, T., et al. (2025). Automated removal of corrupted tilts in cryo-electron tomography. *Journal of Structural Biology: X, 12*, 100130. Elsevier. https://doi.org/10.1016/j.yjsbx.2025.100130

### BibTeX entry

```bibtex
@article{Majtner2025,
  author    = {Majtner, T. and Turoňová, B.},
  title     = {Automated removal of corrupted tilts in cryo-electron tomography},
  journal   = {Journal of Structural Biology: X},
  year      = {2025},
  volume    = {12},
  pages     = {100130},
  doi       = {10.1016/j.yjsbx.2025.100130},
  publisher = {Elsevier}
}
