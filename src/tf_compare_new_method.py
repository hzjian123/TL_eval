from tf_compare_results import *


# 2
# running evaluation
input_lable_path = '/home/fengzhen/fengzhen_ssd/data_set/Featured_HD_2M_1000_data_set/labels'

# input_infer_path = "/home/fengzhen/fengzhen_ssd/data_set_python_script/trafficlight_check_script/data/2022-07-27更换1000帧精选新数据集_冯震输出结果"
input_infer_path = "/home/fengzhen/fengzhen_ssd/data_set_python_script/trafficlight_check_script/data/精选1000帧新数据集_胡佳纯输出/onnx_detection_result"

print("\nSecond step")
print("----------------------------------------------")
# 测试准召率
infer_results(input_lable_path, input_infer_path)
print("----------------------------------------------")
# 打印检测结果转换后的内容
# print(infer_results(input_lable_path, input_infer_path))
