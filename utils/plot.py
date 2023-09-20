from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

def save_plot(R, scenario, save_file, writer):
    first_target = scenario[1]
    x = scenario[1:]
    y = R.loc[first_target].tolist()[1:]

    plt.plot(y)
    plt.xticks(ticks=range(len(x)), labels=x)

    plt.ylim(0, 1)

    plt.title(f"Acc on {first_target} for scenario: {scenario}")
    plt.xlabel("Models")
    plt.ylabel("Acc")

    # Save PNG file
    plt.savefig(save_file)

    # Log to tensorboard
    # # Save this plot to a BytesIO object
    # buffer = BytesIO()
    # plt.savefig(buffer, format='png')
    # plt.close()

    # # Create a PIL Image object
    # buffer.seek(0)
    # image = Image.open(buffer)

    # # Convert the PIL Image to an RGB format if it's RGBA
    # if image.mode == 'RGBA':
    #     image = image.convert('RGB')

    # # Convert the PIL Image to a tensor
    # image_tensor = torch.Tensor(np.array(image)).permute(2, 0, 1)
    # image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension

    # # Log the image to TensorBoard
    # writer.add_image('My_custom_plot', image_tensor, 0)