import threading
import subprocess
from sympy import false

Multi_Thread_Enable = false
# 指定要运行的Python文件和参数
python_file = "/home/yuance/Work/Trianglecounting/directed_global.py"
arguments = [("node", "1", "1000", "1000", "10", "0.3", "0.1"),
             ("node", "2", "1000", "1000", "100", "0.3", "0.1"),
             ("node", "3", "1000", "1000", "1000", "0.3", "0.1"),
             ("node", "4", "1000", "1000", "10000", "0.3", "0.1"),
             ("node", "5", "1000", "1000", "100000", "0.3", "0.1"),

             ("node", "6", "1000", "100", "100", "0.3", "0.1"),
             ("node", "7", "1000", "1000", "1000", "0.3", "0.1"),
             ("node", "8", "1000", "10000", "10000", "0.3", "0.1"),
             ("node", "9", "1000", "100000", "100000", "0.3", "0.1"),

             ("node", "10", "100", "100", "100", "0.3", "0.1"),
             ("node", "11", "1000", "1000", "1000", "0.3", "0.1"),
             ("node", "12", "10000", "10000", "10000", "0.3", "0.1"),
             ("node", "13", "100000", "100000", "100000", "0.3", "0.1"),

             ("node", "14", "10000", "1000", "100", "0.3", "0.1"),
             ("node", "15", "100000", "10000", "1000", "0.3", "0.1"),

             ("prob", "1", "1000", "1000", "1000", "0.1", "0.1"),
             ("prob", "2", "1000", "1000", "1000", "0.3", "0.1"),
             ("prob", "3", "1000", "1000", "1000", "0.5", "0.1"),
             ("prob", "4", "1000", "1000", "1000", "0.7", "0.1"),
             ("prob", "5", "1000", "1000", "1000", "0.9", "0.1"),

             ("prob", "6", "1000", "1000", "1000", "0.3", "0.3"),
             ("prob", "7", "1000", "1000", "1000", "0.3", "0.5"),
             ("prob", "8", "1000", "1000", "1000", "0.3", "0.7"),
             ("prob", "9", "1000", "1000", "1000", "0.3", "0.9")]

def run_script(*args):
    _type, idx, arg1, arg2, arg3, arg4, arg5 = args
    # 打开一个独立的输出文件，以避免多个线程之间的竞争
    output_file = f"/home/yuance/Work/Trianglecounting/output/data/directed_global/{_type}_{idx}.txt"
    with open(output_file, "w") as f:
        # 构造命令行
        command = ["python3", python_file, arg1, arg2, arg3, arg4, arg5]
        # 运行命令，并将输出写入输出文件
        subprocess.run(command, stdout=f)

if Multi_Thread_Enable:
    # 创建线程池并启动线程
    threads = []
    for argument in arguments:
        t = threading.Thread(target=run_script, args=argument)
        threads.append(t)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()
else:
    for argument in arguments:
        run_script(*argument)
