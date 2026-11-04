import pathlib
from pdf2image import convert_from_path


def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images


def save_pdf_as_images(pdf_path):
    subfolder = "images"
    pdf_path = pathlib.Path(pdf_path)
    folder = pdf_path.parent
    stem = pdf_path.stem
    image_folder = folder / subfolder
    image_folder.mkdir(exist_ok=True)
    images = pdf_to_images(pdf_path)
    for i, image in enumerate(images):
        image.save(image_folder / f"{stem}_{i}.png")


if __name__ == "__main__":
    base_path = "/home/brani/code/vlm_training/dataset"
    pdf_files = list(pathlib.Path(base_path).glob("*.pdf"))
    for i, pdf_file in enumerate(pdf_files):
        print(i, pdf_file)
        save_pdf_as_images(pdf_file)
        