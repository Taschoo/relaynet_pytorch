import matplotlib.pyplot as plt

def visualize_segmentation(image, mask, prediction):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(mask, cmap='jet')
    axes[1].set_title('Ground Truth Mask')
    axes[2].imshow(prediction, cmap='jet')
    axes[2].set_title('Predicted Mask')
    plt.show()
