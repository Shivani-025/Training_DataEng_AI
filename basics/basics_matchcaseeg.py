print('Hi')

ch = float(input("Enter a num: "))

match(ch):
    case 1.1:
        print('One')
    case 2.1:
        print('Two')
    case _:
        print('None')