# tensorrt attempt: adds an input cast to the opset-13 float32 onnx graph
# runs on: host pc
# output: model_run3_fp32_op13_cast.onnx

import onnx
from onnx import helper, TensorProto

model = onnx.load('/home/eagle/THESIS/jetson_nano/convtoonnx/model_run3_fp32_op13.onnx')

# change input type from UINT8 to FLOAT
for inp in model.graph.input:
    if inp.type.tensor_type.elem_type == TensorProto.UINT8:
        inp.type.tensor_type.elem_type = TensorProto.FLOAT
        print(f'changed input {inp.name} from UINT8 to FLOAT')

# insert Cast node at the beginning
cast_node = helper.make_node(
    'Cast',
    inputs=['serving_default_images:0'],
    outputs=['input_float'],
    to=TensorProto.FLOAT
)

# remap first node's input
for node in model.graph.node:
    for i, inp in enumerate(node.input):
        if inp == 'serving_default_images:0':
            node.input[i] = 'input_float'
            break
    break

model.graph.node.insert(0, cast_node)

onnx.save(model, '/home/eagle/THESIS/jetson_nano/convtoonnx/model_run3_fp32_op13_cast.onnx')
print('saved model_run3_fp32_op13_cast.onnx')
