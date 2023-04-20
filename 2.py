union_edges = [0, 0, 0, 0, 0, 1]
type_flag = 0
for index, data in enumerate(union_edges):
    type_flag += data << index
print(type_flag)