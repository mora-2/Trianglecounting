
import time
import array

# while True:
    # my_array = array.array('i', [0] * 1e9)
print("start.")
my_array1 = array.array('i', [0] * 1000000000)
print("create my_array1.")
my_array2 = array.array('i', [0] * 1000000000)
print("create my_array2.")
my_array3 = array.array('i', [0] * 1000000000)
print("create my_array3.")
time.sleep(10)  # 延时 3 秒
print("1")
del my_array1
time.sleep(10)  # 延时 3 秒
print("2")
del my_array2
time.sleep(10)  # 延时 3 秒
print("3")


