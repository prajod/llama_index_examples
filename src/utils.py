import os
import fitz
from PIL import Image
import matplotlib.pyplot as plt

def convert_pdf_to_images(pdf_path, output_dir):
    """
    Converts each page of a PDF file to an image and saves it to the output directory.

    Args:
        pdf_path (str): Path to the PDF file.
        output_dir (str): Directory to save the images.

    Returns:
        list: List of paths to the saved images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_document = fitz.open(pdf_path)
    image_paths = []

    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        pix = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        image_filename = f"page_{page_number + 1}.png"
        image_path = os.path.join(output_dir, image_filename)
        image.save(image_path)
        image_paths.append(image_path)

    pdf_document.close()
    return image_paths

def plot_images(image_paths, max_images=9):
    """
    Plots a grid of images from the given paths.

    Args:
        image_paths (list): List of image file paths.
        max_images (int): Maximum number of images to display.
    """
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(3, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= max_images:
                break
    plt.show()
