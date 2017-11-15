#http://python.jobbole.com/86465/
#http://www.jb51.net/article/86766.htm

name = "lzl"


def f1():
    print(name)


def f2():
    name = "eric"
    f1()


f2()