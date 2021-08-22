from Generator import Generate
from PIL import Image
import matplotlib.pyplot as plt

def view_input(n_images = 1):
    """
    Visualize the generated images from Generator. 
    """
    gen = Generate("color-gradient.jpg")
    while(n_images):
        n_images-=1
        fig = plt.imshow(Image.fromarray(next(gen),"RGB"))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()