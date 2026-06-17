# array and linked list
# linked list can be sparse in memory (block do not have to follow itself)

# array : O(1), insertion O(n), deletion O(n)
# linked list : reading O(n), insertion O(1), deletion O(1)


artist = {'radiohead':1,'blur':50,'maria':5,'jul':47}


def find_smallest(list):
    smallest = list[0]
    index = 0
    for key, value in enumerate(list):
        if value < smallest:
            smallest = value
            index=key
    return index, list[index]

def selection_sort(list):
    sorted_array=[]
    for _ in range(len(list)):
        index, smallest = find_smallest(list)
        list.pop(index)
        sorted_array.append(smallest)
    return sorted_array
        


result=selection_sort([-23,1,2,3,4,5,6,23,27,28,33,-70,12313,1441421,412412,43132132424,41242143141242141])
print(result)