from tf_compare_results import *


# 2
# running evaluation
input_lable_path = '/home/fengzhen/work/data_set/Featured_HD_2M_1000_data_set/labels'

input_infer_path = "/home/fengzhen/work/data_set_python_script/trafficlight_check_script/data/2022_10_19benchmark测试，cla_xmt_2_v0810_v1_8_2022_10_17_e6f418/det_output"
# input_infer_path = "/home/fengzhen/work/data_set_python_script/trafficlight_check_script/data/2022_09_09benckmark测试，det_xmt_2_v1_update_2022_08_25_v1_8__346e3/onnx_output_hjc"

print("\nSecond step")
print("----------------------------------------------")
# 测试准召率
infer_results(input_lable_path, input_infer_path)
print("----------------------------------------------")
# 打印检测结果转换后的内容
# print(infer_results(input_lable_path, input_infer_path))
