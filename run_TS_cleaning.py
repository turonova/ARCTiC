import torch
import torch.nn as nn
import timm
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
from torchvision import transforms, models
from cryocat import cryomap, mdoc
from PIL import Image

# -----------------------------
# Parse command-line arguments
# -----------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Clean Cryo-ET Tilt Series with Dual-Direction Deep Learning Thresholding.")

    parser.add_argument('--input_ts', type=str, required=True, help="Input .mrc tilt series")
    parser.add_argument('--cleaned_ts', type=str, required=True, help="Output cleaned .mrc tilt series")
    parser.add_argument('--angle_start', type=float, required=True, help="Start tilt angle")
    parser.add_argument('--angle_step', type=float, required=True, help="Angle step per slice")
    parser.add_argument('--pdf_output', type=str, default="output_visualization.pdf", help="Output PDF visualization")
    parser.add_argument('--csv_output', type=str, default="results.csv", help="Output CSV with classification")
    parser.add_argument('--model', type=str, required=True, help="Path to model .pth file")
    parser.add_argument('--mdoc_input', type=str, help="Optional: Path to input .mdoc file")
    parser.add_argument('--mdoc_output', type=str, help="Optional: Path to output cleaned .mdoc file")
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help="Probability threshold to keep a tilt (0.0 to 1.0). Higher values exclude more tilts.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for GPU inference acceleration")

    return parser.parse_args()

args = parse_arguments()

# -----------------------------
# Assign arguments to variables
# -----------------------------
INPUT_TS = args.input_ts
CLEANED_TS = args.cleaned_ts
ANGLE_START = args.angle_start
ANGLE_STEP = args.angle_step
PDF_OUTPUT = args.pdf_output
CSV_OUTPUT = args.csv_output
MODEL = args.model
CONFIDENCE_THRESHOLD = args.confidence_threshold
BATCH_SIZE = args.batch_size

# -----------------------------
# Setup device (GPU if available)
# -----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dynamic Class Detection & Model Selection
# -----------------------------
state_dict = torch.load(MODEL, map_location="cpu")

num_classes = 2  # Default fallback
for key in state_dict.keys():
    if key in ['fc.weight', 'classifier.1.weight', 'head.fc.weight', 'head.weight']:
        num_classes = state_dict[key].shape[0]
        break

print(f"Detected model configuration: {num_classes} classes.")

def modify_resnet(num_classes):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def modify_efficientnet(num_classes):
    model = models.efficientnet_b3(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

model_mapping = {
    'swin_tiny': lambda: timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes),
    'swin_large': lambda: timm.create_model('swin_large_patch4_window7_224', pretrained=False, num_classes=num_classes),
    'resnet': lambda: modify_resnet(num_classes),
    'efficientnet': lambda: modify_efficientnet(num_classes),
}

for key in model_mapping:
    if key in MODEL:
        model = model_mapping[key]()
        break
else:
    raise ValueError("MODEL must contain 'swin_tiny', 'swin_large', 'resnet', or 'efficientnet'")

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print("Model loaded successfully.")

# -----------------------------
# Image preprocessing
# -----------------------------
size = (320, 320) if 'efficientnet' in MODEL else (224, 224)
image_transforms = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
])

# -----------------------------
# Load tilt series
# -----------------------------
mrc = cryomap.read(INPUT_TS)
num_tilts = mrc.shape[2]

tomo3d = []
class_0_info = []  # Excluded tilts
class_1_info = []  # Kept tilts
csv_data = []

# -----------------------------
# Accelerated Batched Inference Pipeline
# -----------------------------
print(f"Preprocessing and packing {num_tilts} tilts into batches (Size: {BATCH_SIZE})...")

all_predicted_classes = []
all_probabilities = []

tensor_list = []
for i in range(num_tilts):
    image_b16 = cryomap.scale(mrc[:, :, i], 0.0625)
    image_b16 = ((image_b16 - image_b16.min()) * (255.0 / (image_b16.max() - image_b16.min()))).astype('uint8')
    img_pil = Image.fromarray(image_b16).convert("RGB")
    tensor_list.append(image_transforms(img_pil))

for batch_idx in range(0, num_tilts, BATCH_SIZE):
    batch_tensors = torch.stack(tensor_list[batch_idx : batch_idx + BATCH_SIZE]).to(device)
    
    with torch.no_grad():
        output = model(batch_tensors)
        batch_probs = torch.softmax(output, dim=1).cpu().numpy()

    for local_idx, probabilities in enumerate(batch_probs):
        global_idx = batch_idx + local_idx
        
        # Track probability of Class 1 (Good) vs Class 0 (Corrupted)
        prob_corrupted = probabilities[0]
        prob_good = probabilities[1]

        # Dual direction thresholding condition check
        if prob_good < CONFIDENCE_THRESHOLD:
            # Exclude tilt (Its good probability falls below threshold requirements)
            predicted_class = 0
            class_0_info.append((global_idx, prob_corrupted))
        else:
            # Keep tilt (Meets or exceeds goodness target criteria)
            predicted_class = 1
            class_1_info.append((global_idx, prob_good))

        all_predicted_classes.append(predicted_class)
        all_probabilities.append(probabilities)

# -----------------------------
# PDF and Metadata Generation
# -----------------------------
with PdfPages(PDF_OUTPUT) as pdf:
    # --- PAGE 1: Geometric Tilt Angle Plot ---
    fig = plt.figure(figsize=(5, 5))
    plt.axis('off')

    print("Generating validation visualizations...")

    for i in range(num_tilts):
        angle = ANGLE_START + i * ANGLE_STEP
        predicted_class = all_predicted_classes[i]
        probs = all_probabilities[i]

        angle_rad = np.radians(angle)
        
        if predicted_class == 1:
            tomo3d.append(mrc[:, :, i])
            plt.plot([-np.cos(angle_rad), np.cos(angle_rad)], [-np.sin(angle_rad), np.sin(angle_rad)], color='black')
        else:
            plt.plot([-np.cos(angle_rad), np.cos(angle_rad)], [-np.sin(angle_rad), np.sin(angle_rad)], color='red', linestyle='--')
            plt.text(np.cos(angle_rad) * 1.01, np.sin(angle_rad) * 1.09, str(i), fontsize=5, color='red')

        csv_data.append({
            "CurrentIndex": i,
            "ToBeRemoved": (predicted_class == 0),
            "GoodProbability": np.round(probs[1], decimals=4),
            "CorruptProbability": np.round(probs[0], decimals=4),
            "Removed": False
        })

    fig.text(0.5, 0.95, f"Tilt Angle Visualization (Threshold={CONFIDENCE_THRESHOLD})", ha='center', fontsize=14, weight='bold')
    pdf.savefig()
    plt.close()

    # --- PAGE 2: Excluded Tilts Overview ---
    num_excluded = len(class_0_info)
    if num_excluded > 0:
        cols = 3
        rows = (num_excluded // cols) + (num_excluded % cols > 0)
        fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 3))
        axes = np.atleast_1d(axes).flatten()
        fig.subplots_adjust(top=0.8, hspace=0.5, wspace=0.5)

        for i, (index, prob) in enumerate(class_0_info):
            image_b16 = cryomap.scale(mrc[:, :, index], 0.0625)
            image_b16 = ((image_b16 - image_b16.min()) * (255.0 / (image_b16.max() - image_b16.min()))).astype('uint8')
            image_b16 = Image.fromarray(image_b16)

            ax = axes[i]
            ax.imshow(image_b16, cmap='gray')
            ax.axis('off')

            colors = ['red'] * int(prob * 100) + ['black'] * (100 - int(prob * 100))
            discrete_cmap = ListedColormap(colors)

            cbar = fig.colorbar(
                plt.cm.ScalarMappable(cmap=discrete_cmap, norm=plt.Normalize(vmin=0, vmax=1)),
                ax=ax, orientation='vertical', fraction=0.046, pad=0.04
            )
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels(['0%', '50%', '100%'])
            ax.set_title(f"Index: {index} | Corrupt Prob: {prob:.2%}")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.text(0.5, 0.92, "Excluded Tilt Images (Fell Below Goodness Threshold)", ha='center', fontsize=14, weight='bold')
        pdf.savefig()
        plt.close()

# -----------------------------
# Save cleaned .mrc volume
# -----------------------------
if len(tomo3d) == 0:
    print("Warning: All tilt images were excluded. No cleaned tilt series was saved.")
else:
    print("Saving cleaned tilt series...")
    tomo3d = np.stack(tomo3d, axis=2)
    cryomap.write(tomo3d, CLEANED_TS, data_type=np.single)
    print("Saved to:", CLEANED_TS)

# -----------------------------
# Save classification results to CSV
# -----------------------------
df = pd.DataFrame(csv_data)
df.to_csv(CSV_OUTPUT, index=False)
print("Saved classification results to:", CSV_OUTPUT)

# -----------------------------
# Optional: Clean .mdoc file
# -----------------------------
if args.mdoc_input and args.mdoc_output:
    if len(tomo3d) == 0:
        print("No images kept. Skipping .mdoc cleanup.")
    else:
        print("Cleaning associated .mdoc file...")
        df = pd.read_csv(CSV_OUTPUT)
        indices_to_remove = df[df["ToBeRemoved"] == True]["CurrentIndex"].tolist()

        my_mdoc = mdoc.Mdoc(args.mdoc_input)
        my_mdoc.remove_images(indices_to_remove, kept_only=False)
        my_mdoc.write(out_path=args.mdoc_output, overwrite=True)
        print("Cleaned .mdoc file saved to:", args.mdoc_output)

print("Process completed successfully.")