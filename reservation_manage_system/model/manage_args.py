from optparse import OptionParser


def manage_args():
    usage = 'usage %prog (option) type'
    parser = OptionParser(usage=usage)
    parser.add_option('-t', '--type', action='store', type='int', dest='type',
                      help='enter value of type. 1: check, 2: view_all, 3: reservation')
    optins, args = parser.parse_args()
    return optins
