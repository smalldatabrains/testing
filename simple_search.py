def simple_search(list, target):
    counter = 0
    for i in list :
        if i==target:
            return counter
        counter +=1

result = simple_search([-23,1,2,3,4,5,6,23,27,28,33,12313,1441421,412412,43132132424,41242143141242141],23)
print(result)


# this is linear search with complexity O(n)