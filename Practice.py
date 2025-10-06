# def filter_invalid(data):
#     return[item for item in data if 'name' in item]
# data = [{'id':1 , 'name':'Alice'},{'id':2}]
# result=filter_invalid(data)
# print(result)

annotations = [['cat', 10, 20, 50, 50], ['dog', 30, 40, 60, 60]]
annotation_dicts=[{'label':item[0],'bbox':item[1:]} for item in annotations]

print(annotation_dicts)