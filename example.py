from yolox_backbone import create_model, list_models
import torch

print(list_models())

model_names = list_models()
for model_name in model_names:
    print("model_name: ", model_name)
    model = create_model(model_name=model_name, pretrained=True)

    input_tensor = torch.randn((1, 3, 640, 640))
    fpn_output_tensors = model(input_tensor)

    p3, p4, p5 = fpn_output_tensors
    print("input_tensor.shape: ", input_tensor.shape)
    print("p3.shape: ", p3.shape)
    print("p4.shape: ", p4.shape)
    print("p5.shape: ", p5.shape)
    print("-" * 50)
    