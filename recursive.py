# recursion

def countdown(i):
    # base case
    if i==0:
        print('countdown finished')
    # recursive case
    else:
        print(i)
        countdown(i-1)
    

countdown(10)