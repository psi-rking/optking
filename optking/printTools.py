from __future__ import print_function
### Printing functions.

print_opt = None

from sys import stdout


def cleanPrint(arg):
    print(arg, file=stdout, end='')
    return


#print_opt = cleanPrint


def printInit(printFunction=None):
    global print_opt
    if not printFunction:
        print_opt = cleanPrint
    else:
        print_opt = printFunction
    return


def printMat(M, Ncol=7, title=None):
    if title:
        print_opt(title + '\n')
    for row in range(M.shape[0]):
        tab = 0
        for col in range(M.shape[1]):
            tab += 1
            print_opt(" %10.6f" % M[row, col])
            if tab == Ncol and col != (M.shape[1] - 1):
                print_opt("\n")
                tab = 0
        print_opt("\n")
    return


def printMatString(M, Ncol=7, title=None):
    if title:
        print_opt(title + '\n')
    s = ''
    for row in range(M.shape[0]):
        tab = 0
        for col in range(M.shape[1]):
            tab += 1
            s += " %10.6f" % M[row, col]
            if tab == Ncol and col != (M.shape[1] - 1):
                s += '\n'
                tab = 0
        s += '\n'
    return s


def printArray(M, Ncol=7, title=None):
    if title:
        print_opt(title + '\n')
    tab = 0
    for col, entry in enumerate(M):
        tab += 1
        print_opt(" %10.6f" % M[col])
        if tab == Ncol and col != (len(M) - 1):
            print_opt("\n")
            tab = 0
    print_opt("\n")
    return


def printArrayString(M, Ncol=7, title=None):
    if title:
        print_opt(title + '\n')
    tab = 0
    s = ''
    for i, entry in enumerate(M):
        tab += 1
        s += " %10.6f" % entry
        if tab == Ncol and i != (len(M) - 1):
            s += '\n'
            tab = 0
    s += '\n'
    return s


def printGeomGrad(geom, grad):
    print_opt("\tGeometry and Gradient\n")
    Natom = geom.shape[0]

    for i in range(Natom):
        print_opt("\t%20.10f%20.10f%20.10f\n" % (geom[i, 0], geom[i, 1], geom[i, 2]))
    print_opt("\n")
    for i in range(Natom):
        print_opt("\t%20.10f%20.10f%20.10f\n" % (grad[3 * i + 0], grad[3 * i + 1],
                                                 grad[3 * i + 2]))
