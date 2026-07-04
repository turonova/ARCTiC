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
    parser = argparse.ArgumentParser(description="Clean Cryo-ET Tilt Series with Deep Learning.")

    parser.add_argument('--input_ts', type=str, required=True, help="Input .mrc tilt series")
    parser.add_argument('--cleaned_ts', type=str, required=True, help="Output cleaned .mrc tilt series")
    parser.add_argument('--angle_start', type=float, required=True, help="Start tilt angle")
    parser.add_argument('--angle_step', type=float, required=True, help="Angle step per slice")
    parser.add_argument('--pdf_output', type=str, default="output_visualization.pdf", help="Output PDF visualization")
    parser.add_argument('--csv_output', type=str, default="results.csv", help="Output CSV with classification")
    parser.add_argument('--model', type=str, required=True, help="Path to model .pth file")
    parser.add_argument('--mdoc_input', type=str, help="Optional: Path to input .mdoc file")
    parser.add_argument('--mdoc_output', type=str, help="Optional: Path to output cleaned .mdoc file")
    parser.add_argument('--confidence_threshold', type=float, default=0.0, help="Minimum probability required to use prediction (0.0 to 1.0)")

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

# -----------------------------
# Setup device (GPU if available)
# -----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model selection and loading
# -----------------------------
def modify_resnet():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def modify_efficientnet():
    model = models.efficientnet_b3(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

model_mapping = {
    'swin_tiny': lambda: timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2),
    'swin_large': lambda: timm.create_model('swin_large_patch4_window7_224', pretrained=False, num_classes=2),
    'resnet': lambda: modify_resnet(),
    'efficientnet': lambda: modify_efficientnet(),
}

for key in model_mapping:
    if key in MODEL:
        model = model_mapping[key]()
        break
else:
    raise ValueError("MODEL must contain 'swin_tiny', 'swin_large', 'resnet', or 'efficientnet'")

model.load_state_dict(torch.load(MODEL, map_location=device))
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
# Evaluation function
# -----------------------------
def evaluate_single_image(image_input, index, class_0_info, class_1_info, class_null_info):
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input

    image = image_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

    predicted_class = np.argmax(probabilities)
    max_prob = np.max(probabilities)
    is_low_confidence = max_prob < CONFIDENCE_THRESHOLD

    if is_low_confidence:
        class_null_info.append((index, max_prob))
    elif predicted_class == 0:
        class_0_info.append((index, probabilities[0]))
    else:
        class_1_info.append((index, probabilities[1]))

    return predicted_class, probabilities, is_low_confidence

# -----------------------------
# Load tilt series
# -----------------------------
mrc = cryomap.read(INPUT_TS)
tomo3d = []
class_0_info = []
class_1_info = []
class_null_info = []
csv_data = []

# -----------------------------
# PDF output setup
# -----------------------------
with PdfPages(PDF_OUTPUT) as pdf:
    fig = plt.figure(figsize=(5, 5))
    plt.axis('off')

    print("Processing tilt series...")

    for i in range(mrc.shape[2]):
        angle = ANGLE_START + i * ANGLE_STEP
        image_b16 = cryomap.scale(mrc[:, :, i], 0.0625)
        image_b16 = ((image_b16 - image_b16.min()) * (255.0 / (image_b16.max() - image_b16.min()))).astype('uint8')
        image_b16 = Image.fromarray(image_b16).convert("RGB")

        predicted_class, probs, is_low_confidence = evaluate_single_image(image_b16, i, class_0_info, class_1_info, class_null_info)

        angle_rad = np.radians(angle)
        
        # Logic: Keep if the model explicitly says keep (1) OR if it is low confidence
        if predicted_class == 1 or is_low_confidence:
            tomo3d.append(mrc[:, :, i])
            
            # Use orange for low confidence tilts so they stand out in the visualization
            color = 'orange' if is_low_confidence else 'black'
            plt.plot([-np.cos(angle_rad), np.cos(angle_rad)], [-np.sin(angle_rad), np.sin(angle_rad)], color=color)
            if is_low_confidence:
                plt.text(np.cos(angle_rad) * 1.01, np.sin(angle_rad) * 1.09, f"{i}?", fontsize=5, color='orange')
        else:
            # Drop only if confidently identified as class 0
            plt.plot([-np.cos(angle_rad), np.cos(angle_rad)], [-np.sin(angle_rad), np.sin(angle_rad)], color='red', linestyle='--')
            plt.text(np.cos(angle_rad) * 1.01, np.sin(angle_rad) * 1.09, str(i), fontsize=5, color='red')

        csv_data.append({
            "CurrentIndex": i,
            "ToBeRemoved": (predicted_class == 0) and (not is_low_confidence),
            "Confident": not is_low_confidence,
            "MaxProbability": np.round(np.max(probs), decimals=4),
            "Removed": False
        })

    fig.text(0.5, 0.95, "Tilt Angle Visualization (Orange=Uncertain)", ha='center', fontsize=14, weight='bold')
    pdf.savefig()
    plt.close()

    # Plot probability bars for class 0 (Confident Removals)
    num_images = len(class_0_info)
    if num_images > 0:
        cols = 3
        rows = (num_images // cols) + (num_images % cols > 0)
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

            ax.set_title(f"Index: {index} | Prob: {prob:.2%}")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.text(0.5, 0.9, "Excluded Tilt Images with Probability Scale Bar", ha='center', fontsize=14, weight='bold')
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