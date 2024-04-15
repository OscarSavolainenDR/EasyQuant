import torch
from pathlib import Path

def evaluate(model, device_str="cuda"):

    # Download an example image from the pytorch website
    """
    downloads an image from a PyTorch repository and preprocesses it for feeding
    into a deep learning model. It then moves the input and model to the GPU or
    CPU, runs the model on the input, and produces probabilities for each of
    ImageNet's 1000 categories.

    Args:
        model (nn.Module.): 3D convolutional neural network (CNN) that is being
            evaluated, and it is expected to be a torch.nn.Module object.
            
            		- `device_str`: A string representing the device to move the input
            and model to (either "cpu" or "cuda").
            		- `input_batch`: A tensor of shape 1000, representing a mini-batch
            of input images.
            		- `model`: A PyTorch model that takes the input batch as its argument.
            The model has various properties, such as:
            		+ `to(device_str)`: Moves the model to the specified device (either
            "cpu" or "cuda").
            		+ `unsqueeze(0)`: Adds a dimension of size 1 to the input batch,
            which is required by PyTorch models.
            		+ `dimension` and `dim`: These are not explicitly provided in the
            code snippet, but can be inferred from the context as the dimension
            of the output of the `model` function.
            		+ `torch.nn.functional.softmax(output[0], dim=0)`: Applies a softmax
            activation function to the output of the model, which produces
            probabilities for each class.
        device_str (str): device (either "cpu" or "cuda") to which the input tensor
            and the model will be moved for computation, allowing the function to
            take advantage of GPU acceleration when available.

    """
    import urllib

    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    # sample execution (requires torchvision)
    from PIL import Image
    from torchvision import transforms

    input_image = Image.open(filename)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model

    # Move the input and model to GPU for speed if available, or to CPU if converted
    if not (device_str in ["cpu", "cuda"]):
        raise NotImplementedError("`device_str` should be 'cpu' or 'cuda' ")
    if device_str == "cuda":
        assert torch.cuda.is_available(), "Check CUDA is available"

    input_batch = input_batch.to(device_str)
    model.to(device_str)

    with torch.no_grad():
        output = model(input_batch)

    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
    # print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)

    # Read the categories
    with open(Path(__file__).resolve().parent / "imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print('\nEvaluation:')
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
