

class MyList(list):
    def __init__(self, iterable, name=None, **args):
        list.__init__(self, iterable)
        self.name=name


def main():
    a = MyList([1,2,3,4], name="yeah man")

    print(a.name)
    print(a)
    print(a[0])

if __name__ == "__main__":
    main()