from ImageMerger import Merger, ImageToMerge, MERGE_GRID
import os

def merger_image(num_samples, image_name, image_folder):
    num_samples=num_samples
    list_images = []
    for example in range(num_samples):
        list_images.append(ImageToMerge(path=os.path.join(image_folder, f"ex{example}_{image_name}.png")))
    m = Merger(list_images=list_images, limit_horizontal=10)
    filename = os.path.join(image_folder, f"{image_name}.png")
    m.save_image(filename=filename)
    for example in range(num_samples):
        os.remove(os.path.join(image_folder, f"ex{example}_{image_name}.png"))
