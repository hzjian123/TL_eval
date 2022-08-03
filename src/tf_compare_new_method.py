from tf_compare_results import *


# 2
# running evaluation
input_lable_path = '/home/fengzhen/work/data_set/Featured_HD_2M_1000_data_set/labels'

input_infer_path = "/home/fengzhen/work/data_set_python_script/trafficlight_check_script/data/dec20220625/bin_output"
# input_infer_path = "/home/fengzhen/work/data_set_python_script/trafficlight_check_script/data/dec20220625/trt_output"
# input_infer_path = "/home/fengzhen/work/data_set_python_script/trafficlight_check_script/data/dec20220625/onnx_output_hjc"

print("\nSecond step")
print("----------------------------------------------")
# 测试准召率
infer_results(input_lable_path, input_infer_path)
print("----------------------------------------------")
# 打印检测结果转换后的内容
# print(infer_results(input_lable_path, input_infer_path))
