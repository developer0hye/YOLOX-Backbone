import yolox_backbone
import torch
from pprint import pprint

pprint(yolox_backbone.list_models())

model_names = yolox_backbone.list_models()
for model_name in model_names:
    print("model_name: ", model_name)
    
    model = yolox_backbone.create_model(model_name=model_name, 
                                        pretrained=True, 
                                        out_features=["P3", "P4", "P5"]
                                        )
    model.eval()
    
    input_tensor = torch.randn((1, 3, 640, 640))
    fpn_output_tensors = model(input_tensor)

    p3 = fpn_output_tensors["P3"]
    p4 = fpn_output_tensors["P4"]
    p5 = fpn_output_tensors["P5"]
    
    print("input_tensor.shape: ", input_tensor.shape)
    print("p3.shape: ", p3.shape)
    print("p4.shape: ", p4.shape)
    print("p5.shape: ", p5.shape)
    print("-" * 50)
    