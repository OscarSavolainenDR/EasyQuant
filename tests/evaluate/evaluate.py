import torch
from pathlib import Path

def evaluate(model, device_str="cuda"):

    # Download an example image from the pytorch website
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
    with open(Path("tests" / "evaluate" / "imagenet_classes.txt"), "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
