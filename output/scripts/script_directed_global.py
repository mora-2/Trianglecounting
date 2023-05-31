import threading
import concurrent.futures
import subprocess

max_workers = 1  # 最大并发线程数

# 标准对照组
#  ("node", "11", "1000", "1000", "1000", "0.3", "0.1"),

# 指定要运行的Python文件和参数
python_file = "/home/yuance/Work/Trianglecounting/directed_global.py"
# arguments = [("node", "1", "20", "20", "20", "0.3", "0.1")]
arguments = [
             ("node", "1", "1000", "1000", "10", "0.3", "0.1"),
             ("node", "2", "1000", "1000", "100", "0.3", "0.1"),
             ("node", "3", "1000", "1000", "1000", "0.3", "0.1"),
            #  ("node", "4", "1000", "1000", "10000", "0.3", "0.1"),**********
            #  ("node", "5", "1000", "1000", "100000", "0.3", "0.1"),**********

             ("node", "6", "1000", "100", "100", "0.3", "0.1"),
            #  ("node", "7", "1000", "1000", "1000", "0.3", "0.1"),-----------
            #  ("node", "8", "1000", "10000", "10000", "0.3", "0.1"),*************
            #  ("node", "9", "1000", "100000", "100000", "0.3", "0.1"),**********

             ("node", "10", "100", "100", "100", "0.3", "0.1"),
            #  ("node", "11", "1000", "1000", "1000", "0.3", "0.1"),-----------
            #  ("node", "12", "10000", "10000", "10000", "0.3", "0.1"),**********
            #  ("node", "13", "100000", "100000", "100000", "0.3", "0.1"),**********

            #  ("node", "14", "10000", "1000", "100", "0.3", "0.1"),**********
            #  ("node", "15", "100000", "10000", "1000", "0.3", "0.1"),**********

             ("prob", "1", "1000", "1000", "1000", "0.1", "0.1"),
            #  ("prob", "2", "1000", "1000", "1000", "0.3", "0.1"),-----------
             ("prob", "3", "1000", "1000", "1000", "0.5", "0.1"),
             ("prob", "4", "1000", "1000", "1000", "0.7", "0.1"),
             ("prob", "5", "1000", "1000", "1000", "0.9", "0.1"),

             ("prob", "6", "1000", "1000", "1000", "0.3", "0.3"),
             ("prob", "7", "1000", "1000", "1000", "0.3", "0.5"),
             ("prob", "8", "1000", "1000", "1000", "0.3", "0.7"),
             ("prob", "9", "1000", "1000", "1000", "0.3", "0.9")
             ]

def run_script(args):
    _type, idx, arg1, arg2, arg3, arg4, arg5 = args
    # 打开一个独立的输出文件，以避免多个线程之间的竞争
    output_file = f"/home/yuance/Work/Trianglecounting/output/data/directed_global/{_type}_{idx}.txt"
    with open(output_file, "w") as f:
        # 构造命令行
        command = ["python3", python_file, arg1, arg2, arg3, arg4, arg5]
        # 运行命令，并将输出写入输出文件
        subprocess.run(command, stdout=f)


# 使用ThreadPoolExecutor创建线程池
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # 提交每个任务
    future_to_arg = {executor.submit(run_script, arg): arg for arg in arguments}

    # 等待所有任务完成
    for future in concurrent.futures.as_completed(future_to_arg):
        arg = future_to_arg[future]
        try:
            future.result()  # 获取任务结果
        except Exception as exc:
            print(f"{arg} generated an exception: {exc}")
