import onnx
from onnxsim import simplify

print("obj_pnetcls")
onnx_model = onnx.load("/home/ycb/InstanceFusion/traced/obj_pnetcls")
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph)) # 打印计算图s

model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, "/home/ycb/InstanceFusion/traced/")


print("obj_pnetenc")
onnx_model = onnx.load("/home/ycb/InstanceFusion/traced/obj_pnetenc")
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph)) # 打印计算图s

print("rel_pnetcls")
onnx_model = onnx.load("/home/ycb/InstanceFusion/traced/rel_pnetcls")
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph)) # 打印计算图s

print("rel_pnetenc")
onnx_model = onnx.load("/home/ycb/InstanceFusion/traced/rel_pnetenc")
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph)) # 打印计算图s
