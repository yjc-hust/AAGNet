import os
import win32com.client
import time

def convert_step_to_sldprt(input_folder, output_folder):
    """
    将 STEP 文件批量转换为 SolidWorks (.SLDPRT) 文件
    """
    swApp = win32com.client.Dispatch("SldWorks.Application")
    swApp.Visible = True
    
    # 获取输入文件夹中所有 STEP 文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".STEP"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename.replace(".STEP", ".SLDPRT"))
            
            # 打开 STEP 文件
            model = swApp.OpenDoc6(input_file, 3, 0, "", 0, 0)  # 3: STEP 文件类型
            
            if model is not None:
                # 保存为 SolidWorks 文件 (.SLDPRT)
                model.SaveAs(output_file)
                print(f"文件已保存为 SolidWorks (.SLDPRT): {output_file}")
                time.sleep(1)  # 添加延迟，确保每个文件处理顺利完成
            else:
                print(f"无法打开文件: {input_file}")
    
    print("STEP 文件批量转换为 SLDPRT 完成")

def convert_sldprt_to_step(input_folder, output_folder):
    """
    将 SolidWorks (.SLDPRT) 文件批量转换为 STEP 文件
    """
    swApp = win32com.client.Dispatch("SldWorks.Application")
    swApp.Visible = True
    
    # 获取输入文件夹中所有 SLDPRT 文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".SLDPRT"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename.replace(".SLDPRT", ".STEP"))
            
            # 打开 SLDPRT 文件
            model = swApp.OpenDoc6(input_file, 1, 0, "", 0, 0)  # 1: Part 文件类型 (.SLDPRT)
            
            if model is not None:
                # 将文件保存为 STEP 文件
                model.SaveAs(output_file)
                print(f"文件已保存为 STEP: {output_file}")
                time.sleep(1)  # 添加延迟，确保每个文件处理顺利完成
            else:
                print(f"无法打开文件: {input_file}")
    
    print("SLDPRT 文件批量转换为 STEP 完成")

def main():
    input_folder = r"C:\Users\Lenovo\AAGNet\data\steps1"  # 输入文件夹路径
    output_folder = r"C:\Users\Lenovo\AAGNet\data\sw_steps1"  # 输出文件夹路径
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 执行 STEP 转 SLDPRT 转换
    convert_step_to_sldprt(input_folder, output_folder)
    
    # 执行 SLDPRT 转 STEP 转换
    convert_sldprt_to_step(input_folder, output_folder)

if __name__ == "__main__":
    main()