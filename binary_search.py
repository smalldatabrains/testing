def binary_search (list, target):
    high = len(list) - 1
    low = 0

    print("List has ",high, " elements")

    while low <= high:
        mid = (low + high) //2
        guess = list[mid]
        if guess == target:
            return mid
        if guess>target:
            high = mid - 1 # this is where the conquer part lies
        else:
            low = mid + 1 # this is where the conquer part lies


result = binary_search([-23,1,2,3,4,5,6,23,27,28,33,12313,1441421,412412,43132132424,41242143141242141],23)
print(result)

## log tims complexituy log2(n) operations to perform