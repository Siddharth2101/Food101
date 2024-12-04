def slope_of_cubic(a,x):
    a=str(a)
    a3=float(a[(a.index("(")+1):a.index(",")])
    a=a[(a.index(",")+1):]
    a2=float(a[0:a.index(",")])
    a=a[(a.index(",")+1):]
    a1=float(a[0:a.index(",")])
    x=float(x)
    return 3*a3*x*x+2*a2*x+a1

# print(slope_of_cubic(input("Enter a tuple : "),input("Enter x : ")))

