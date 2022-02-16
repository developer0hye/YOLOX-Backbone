import yolox_backbone
import torch
from pprint import pprint

pprint(yolox_backbone.list_models())

model_names = yolox_backbone.list_models()
for model_name in model_names:
    print("model_name: ", model_name)
    
    model = yolox_backbone.create_model(model_name=model_name, 
                                        pretrained=True, 
                                        out_features=["C3", "C4", "C5"]
                                        )
    model.eval()
    
    input_tensor = torch.randn((1, 3, 640, 640))
    output_tensor = model(input_tensor)
    
    c3 = output_tensor["C3"]
    c4 = output_tensor["C4"]
    c5 = output_tensor["C5"]

    p3 = output_tensor["P3"]
    p4 = output_tensor["P4"]
    p5 = output_tensor["P5"]
    
    print("input_tensor.shape: ", input_tensor.shape)
    print("c3.shape: ", c3.shape)
    print("c4.shape: ", c4.shape)
    print("c5.shape: ", c5.shape)
    print("p3.shape: ", p3.shape)
    print("p4.shape: ", p4.shape)
    print("p5.shape: ", p5.shape)
    print("-" * 50)
    